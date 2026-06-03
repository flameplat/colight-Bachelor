import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python analyze_run.py <path_to_run_dir>")
    print("Example: python analyze_run.py records/pilot_grid/anon_6_6_900_0.3_turn.json_06_03_15_32_40")
    sys.exit(1)

RUN_DIR = sys.argv[1]
OUT_DIR = os.path.join(RUN_DIR, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(RUN_DIR, "train_round")
rounds = sorted([d for d in os.listdir(TRAIN_DIR) if d.startswith("round_")],
                key=lambda x: int(x.split("_")[1]))
print(f"Found {len(rounds)} rounds: {rounds}")

# ── Detect grid dimensions from roadnet ──────────────────────────────────────
first_gen = os.path.join(TRAIN_DIR, rounds[0],
                         sorted([d for d in os.listdir(os.path.join(TRAIN_DIR, rounds[0]))
                                 if d.startswith("generator_")])[0])
with open(os.path.join(first_gen, "cityflow.config")) as f:
    _cfg = json.load(f)
_roadnet_path = os.path.join(_cfg["dir"], _cfg["roadnetFile"])
with open(_roadnet_path) as f:
    _rn = json.load(f)
_real_inters = [i["id"] for i in _rn["intersections"] if not i.get("virtual", True)]
_rows = sorted(set(int(i.split("_")[1]) for i in _real_inters))
_cols = sorted(set(int(i.split("_")[2]) for i in _real_inters))
NUM_ROW = len(_rows)
NUM_COL = len(_cols)
NUM_INTERSECTIONS = NUM_ROW * NUM_COL
print(f"Grid detected: {NUM_ROW} rows x {NUM_COL} cols = {NUM_INTERSECTIONS} intersections")

# ── Collect data across all rounds and generators ────────────────────────────
# queue_data[round_idx][inter_idx] = (times, queues)
# vehicle_dfs: list of dataframes from all rounds/generators

queue_by_round = {}   # round_idx -> np.array shape (NUM_INTERSECTIONS, T)
vehicle_dfs_by_round = {}

for round_name in rounds:
    round_idx = int(round_name.split("_")[1])
    round_dir = os.path.join(TRAIN_DIR, round_name)
    generators = sorted([d for d in os.listdir(round_dir) if d.startswith("generator_")],
                        key=lambda x: int(x.split("_")[1]))

    # aggregate queue across generators (take first generator — one per round in CoLight)
    gen_dir = os.path.join(round_dir, generators[0])

    queues_this_round = []
    times_ref = None
    for i in range(NUM_INTERSECTIONS):
        path = os.path.join(gen_dir, f"inter_{i}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            records = pickle.load(f)
        times, queues = [], []
        for rec in records:
            t = rec["time"]
            stopped = rec["state"]["lane_num_vehicle_been_stopped_thres1"]
            q = sum(stopped) if stopped else 0
            times.append(t)
            queues.append(q)
        if times_ref is None:
            times_ref = times
        queues_this_round.append(queues)

    queue_by_round[round_idx] = (times_ref, np.array(queues_this_round))  # (T,), (N, T)

    # vehicle data
    dfs = []
    for i in range(NUM_INTERSECTIONS):
        path = os.path.join(gen_dir, f"vehicle_inter_{i}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if dfs:
        merged = pd.concat(dfs).drop_duplicates(subset=["Unnamed: 0"])
        merged.columns = ["vehicle_id", "enter_time", "leave_time"]
        merged["round"] = round_idx
        vehicle_dfs_by_round[round_idx] = merged

print(f"Loaded data for rounds: {list(queue_by_round.keys())}")

# ── Combined vehicle dataframe ────────────────────────────────────────────────
all_vehicles = pd.concat(vehicle_dfs_by_round.values())
completed = all_vehicles.dropna(subset=["leave_time"]).copy()
completed["travel_time"] = completed["leave_time"] - completed["enter_time"]

# ── PLOT 1: Mean queue over time per round ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for round_idx, (times, qmat) in sorted(queue_by_round.items()):
    mean_q = qmat.mean(axis=0)
    ax.plot(times, mean_q, label=f"Round {round_idx}", alpha=0.8)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Avg stopped vehicles per intersection")
ax.set_title("Mean queue over time — all rounds")
ax.legend(fontsize=7, ncol=5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "queue_over_time_by_round.png"), dpi=150)
plt.close()
print("Saved queue_over_time_by_round.png")

# ── PLOT 2: Heatmap — avg queue per intersection (last round) ─────────────────
last_round = max(queue_by_round.keys())
_, qmat_last = queue_by_round[last_round]
avg_queue = qmat_last.mean(axis=1)
grid = avg_queue.reshape(NUM_ROW, NUM_COL)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Avg stopped vehicles")
ax.set_xticks(range(NUM_COL))
ax.set_xticklabels([f"col {j+1}" for j in range(NUM_COL)])
ax.set_yticks(range(NUM_ROW))
ax.set_yticklabels([f"row {i+1}" for i in range(NUM_ROW)])
for i in range(NUM_ROW):
    for j in range(NUM_COL):
        ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title(f"Avg queue per intersection (round {last_round})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "queue_heatmap_last_round.png"), dpi=150)
plt.close()
print("Saved queue_heatmap_last_round.png")

# ── PLOT 3: Throughput per 60s window (last round) ────────────────────────────
last_round_vehicles = vehicle_dfs_by_round[last_round].dropna(subset=["leave_time"]).copy()
last_round_vehicles["travel_time"] = last_round_vehicles["leave_time"] - last_round_vehicles["enter_time"]
max_time = int(last_round_vehicles["leave_time"].max())
bins = np.arange(0, max_time + 60, 60)
throughput, bin_edges = np.histogram(last_round_vehicles["leave_time"], bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(bin_centers, throughput, width=55, color="seagreen", alpha=0.8)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Vehicles completed")
ax.set_title(f"Throughput per 60s window (round {last_round})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "throughput_last_round.png"), dpi=150)
plt.close()
print("Saved throughput_last_round.png")

# ── PLOT 4: Travel time distribution (last round) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(last_round_vehicles["travel_time"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(last_round_vehicles["travel_time"].mean(), color="red", linestyle="--",
           label=f"Mean: {last_round_vehicles['travel_time'].mean():.1f}s")
ax.axvline(last_round_vehicles["travel_time"].median(), color="orange", linestyle="--",
           label=f"Median: {last_round_vehicles['travel_time'].median():.1f}s")
ax.set_xlabel("Travel time (s)")
ax.set_ylabel("Count")
ax.set_title(f"Travel time distribution (round {last_round})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "travel_time_dist_last_round.png"), dpi=150)
plt.close()
print("Saved travel_time_dist_last_round.png")

# ── PLOT 5: Queue per intersection over time — 6x6 small multiples (last round)
_, qmat_last = queue_by_round[last_round]
times_last = queue_by_round[last_round][0]
fig, axes = plt.subplots(NUM_ROW, NUM_COL, figsize=(18, 12), sharex=True, sharey=True)
for i in range(NUM_ROW):
    for j in range(NUM_COL):
        idx = i * NUM_COL + j
        axes[i][j].plot(times_last, qmat_last[idx], linewidth=0.8, color="steelblue")
        axes[i][j].set_title(f"{i+1},{j+1}", fontsize=7)
        axes[i][j].tick_params(labelsize=6)
fig.suptitle(f"Queue per intersection over time (round {last_round})", fontsize=13)
fig.text(0.5, 0.02, "Simulation time (s)", ha="center")
fig.text(0.02, 0.5, "Stopped vehicles", va="center", rotation="vertical")
plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
plt.savefig(os.path.join(OUT_DIR, "queue_per_intersection_last_round.png"), dpi=150)
plt.close()
print("Saved queue_per_intersection_last_round.png")

# ── PLOT 6: Avg travel time per round (learning curve) ───────────────────────
round_stats = []
for r, df in sorted(vehicle_dfs_by_round.items()):
    comp = df.dropna(subset=["leave_time"]).copy()
    comp["travel_time"] = comp["leave_time"] - comp["enter_time"]
    total = len(df)
    done = len(comp)
    round_stats.append({
        "round": r,
        "avg_travel_time": comp["travel_time"].mean(),
        "throughput_pct": done / total * 100
    })
stats_df = pd.DataFrame(round_stats)

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
ax1.plot(stats_df["round"], stats_df["avg_travel_time"], "o-", color="steelblue", label="Avg travel time")
ax2.plot(stats_df["round"], stats_df["throughput_pct"], "s--", color="seagreen", label="Throughput %")
ax1.set_xlabel("Round")
ax1.set_ylabel("Avg travel time (s)", color="steelblue")
ax2.set_ylabel("Throughput (%)", color="seagreen")
ax1.set_title("Learning curve: travel time & throughput per round")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learning_curve.png"), dpi=150)
plt.close()
print("Saved learning_curve.png")

# ── Summary ───────────────────────────────────────────────────────────────────
last_stats = stats_df[stats_df["round"] == last_round].iloc[0]
print("\n── Summary (last round) ─────────────────────")
total = len(vehicle_dfs_by_round[last_round])
done = len(last_round_vehicles)
print(f"Total vehicles spawned : {total}")
print(f"Completed trips        : {done} ({last_stats['throughput_pct']:.1f}%)")
print(f"Avg travel time        : {last_stats['avg_travel_time']:.1f}s")
print(f"Median travel time     : {last_round_vehicles['travel_time'].median():.1f}s")
print(f"Avg queue (all inters) : {qmat_last.mean():.2f} vehicles")
print(f"Peak queue (any inter) : {qmat_last.max():.0f} vehicles")
most_congested = avg_queue.argmax()
print(f"Most congested inter   : row {most_congested//NUM_COL+1}, col {most_congested%NUM_COL+1}")
print(f"Outputs saved to       : {OUT_DIR}/")

# ── Turn movement congestion — all intersections, last round ──────────────────
# lane_num_vehicle_been_stopped_thres1 order: W(0-2), E(3-5), N(6-8), S(9-11)
# within each approach: lane 0=left, lane 1=straight, lane 2=right
gen_dir_last = os.path.join(TRAIN_DIR, f"round_{last_round}",
                            sorted([d for d in os.listdir(os.path.join(TRAIN_DIR, f"round_{last_round}"))
                                    if d.startswith("generator_")])[0])
approach_labels = ["W", "E", "N", "S"]
move_labels     = ["left", "straight", "right"]

# aggregate per-intersection turn movement averages
inter_turn_avg = {}  # inter_idx -> {(approach, move): avg_stopped}
for idx in range(NUM_INTERSECTIONS):
    path = os.path.join(gen_dir_last, f"inter_{idx}.pkl")
    with open(path, "rb") as f:
        records = pickle.load(f)
    totals = {}
    for ap_i, ap in enumerate(approach_labels):
        for mv_i, mv in enumerate(move_labels):
            lane_idx = ap_i * 3 + mv_i
            vals = [r["state"]["lane_num_vehicle_been_stopped_thres1"][lane_idx]
                    for r in records if r["state"]["lane_num_vehicle_been_stopped_thres1"]]
            totals[(ap, mv)] = np.mean(vals) if vals else 0
    inter_turn_avg[idx] = totals

# find top 5 most congested approach+movement combinations across all intersections
all_movements = []
for idx, totals in inter_turn_avg.items():
    row = idx // NUM_COL + 1
    col = idx % NUM_COL + 1
    for (ap, mv), val in totals.items():
        all_movements.append((val, row, col, ap, mv))
all_movements.sort(reverse=True)

print("\n── Top 10 Most Congested Approach+Movement Combinations ──")
print(f"  {'Inter':>8}  {'Approach':>8}  {'Movement':>10}  {'Avg Stopped':>12}")
for val, row, col, ap, mv in all_movements[:10]:
    print(f"  ({row},{col}){' ':>4}  {ap:>8}  {mv:>10}  {val:>12.1f}")

# per-intersection breakdown for the most congested intersection
mc_row = most_congested // NUM_COL + 1
mc_col = most_congested % NUM_COL + 1
print(f"\n── Turn Breakdown at Most Congested Intersection ({mc_row},{mc_col}) ──")
totals = inter_turn_avg[most_congested]
for ap in approach_labels:
    parts = "  ".join(f"{mv}={totals[(ap, mv)]:.1f}" for mv in move_labels)
    print(f"  {ap} approach: {parts}")

# ── Corridor congestion analysis (last round) ─────────────────────────────────
corridor_h = {}  # row -> avg queue on E-W approaches
corridor_v = {}  # col -> avg queue on N-S approaches

for row in range(NUM_ROW):
    ew_queues = []
    for col in range(NUM_COL):
        idx = row * NUM_COL + col
        path = os.path.join(gen_dir_last, f"inter_{idx}.pkl")
        with open(path, "rb") as f:
            records = pickle.load(f)
        for rec in records:
            stopped = rec["state"]["lane_num_vehicle_been_stopped_thres1"]
            if stopped:
                ew_queues.append(sum(stopped[0:3]) + sum(stopped[3:6]))  # W + E lanes
    corridor_h[row + 1] = np.mean(ew_queues) if ew_queues else 0

for col in range(NUM_COL):
    ns_queues = []
    for row in range(NUM_ROW):
        idx = row * NUM_COL + col
        path = os.path.join(gen_dir_last, f"inter_{idx}.pkl")
        with open(path, "rb") as f:
            records = pickle.load(f)
        for rec in records:
            stopped = rec["state"]["lane_num_vehicle_been_stopped_thres1"]
            if stopped:
                ns_queues.append(sum(stopped[6:9]) + sum(stopped[9:12]))  # N + S lanes
    corridor_v[col + 1] = np.mean(ns_queues) if ns_queues else 0

print("\n── Corridor Congestion (last round) ─────────────")
print("Horizontal corridors (E-W traffic), sorted by congestion:")
for row, val in sorted(corridor_h.items(), key=lambda x: -x[1]):
    bar = "█" * int(val / 2)
    print(f"  Row {row}: {val:5.1f} avg stopped  {bar}")

print("\nVertical corridors (N-S traffic), sorted by congestion:")
for col, val in sorted(corridor_v.items(), key=lambda x: -x[1]):
    bar = "█" * int(val / 2)
    print(f"  Col {col}: {val:5.1f} avg stopped  {bar}")
