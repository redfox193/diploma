import json
from collections import defaultdict

with open("info.json", "r") as f:
    info = json.load(f)

with open("arrivals.json", "r") as f:
    lines = f.readlines()
    
schedule_raw = []
for line in lines:
    arrival = json.loads(line)
    line_name = arrival["thread_id"]
    departure_tm = arrival["departure_tm"]
    schedule_raw.append({
        "line_name": line_name,
        "time": departure_tm,
    })

lines_info = info["lines"]
control_points_info = info["control_points"]

line_name_to_id_and_cp_id_from = {line["name"]: (line["id"], line["cp_id_from"]) for line in lines_info}

schedule_map = defaultdict(list)
for row in schedule_raw:
    line_name = row["line_name"]
    time = row["time"]

    line_id, line_cp_id_from = line_name_to_id_and_cp_id_from[line_name]
    schedule_map[line_cp_id_from].append({
        "line_id": line_id,
        "time": time if time > 240 else time + 1440,
    })

schedule = []
for cp_id, entries in schedule_map.items():
    entries = sorted(entries, key=lambda x: x["time"])

    schedule.append({
        "cp_id": cp_id,
        "departures": entries,
    })

info.update({"schedule": schedule})

with open("schedule.json", "w") as f:
    json.dump(info, f, indent=2)