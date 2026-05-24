from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import PPOAgentConfig


class ActorCritic(nn.Module):
    """Actor-Critic сеть с общим backbone."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_layers: List[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()

        activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers: List[nn.Module] = []
        in_size = obs_size
        for h_size in hidden_layers:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(activation_fn())
            in_size = h_size
        self.state_net = nn.Sequential(*layers)

        self.actor = nn.Linear(in_size, action_size)
        self.critic = nn.Linear(in_size, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.state_net(obs)
        return self.actor(latent), self.critic(latent).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    """Буфер для хранения on-policy данных."""

    def __init__(self) -> None:
        self.data: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)


class PPOAgent:
    """PPO-агент для задачи составления графиков смен водителей."""

    def __init__(
        self, config: PPOAgentConfig, state_dim: int, action_dim: int
    ) -> None:
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(
            obs_size=state_dim,
            action_size=action_dim,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Выбрать действие с маскированием недопустимых действий."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.policy(obs_t.unsqueeze(0))
        logits = logits.squeeze(0)

        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        if torch.sum(mask_t).item() == 0:
            mask_t = torch.ones_like(mask_t)
        masked_logits = logits + torch.log(mask_t + 1e-8)

        dist = Categorical(logits=masked_logits)
        if deterministic:
            action = torch.argmax(dist.probs).item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        return action, log_prob, value.item()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO update с GAE. Возвращает метрики обучения."""
        obs = torch.as_tensor(
            np.stack([t.obs for t in buffer.data]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [t.action for t in buffer.data],
            dtype=torch.int64,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            [t.log_prob for t in buffer.data],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = [t.reward for t in buffer.data]
        values = [t.value for t in buffer.data]
        dones = [t.done for t in buffer.data]

        returns, advantages = self._compute_gae(rewards, values, dones)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        dataset = torch.utils.data.TensorDataset(
            obs, actions, old_log_probs, returns_t, advantages_t
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
            drop_last=False,
        )

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []

        for _ in range(self.config.epochs):
            for batch in loader:
                b_obs, b_act, b_old_lp, b_ret, b_adv = batch

                logits, values_pred = self.policy(b_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_act)

                ratios = torch.exp(log_probs - b_old_lp)
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                policy_loss = -torch.min(ratios * b_adv, clipped * b_adv).mean()

                value_loss = nn.functional.mse_loss(values_pred, b_ret)
                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                approx_kls.append((b_old_lp - log_probs).mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def save(self, path: str) -> None:
        """Сохранить веса модели."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Загрузить веса модели."""
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """Generalized Advantage Estimation."""
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.config.gamma * next_value * mask - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns, advantages
