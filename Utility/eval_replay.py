"""Run one evaluation episode from a saved checkpoint and write replay files."""
import os
import sys
import json

# Always run from project root regardless of where this script is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import model_test

MODEL_DIR = "model/6_6_900_turn_drain_150_optimal/anon_6_6_900_0.3_turn_drain.json_06_04_07_56_35"
RECORDS_DIR = MODEL_DIR.replace("model", "records")
ROUND = 110
RUN_COUNTS = 3600

with open(os.path.join(RECORDS_DIR, "traffic_env.conf"), "r") as f:
    dic_traffic_env_conf = json.load(f)

dic_traffic_env_conf["SAVEREPLAY"] = True

model_test.test(MODEL_DIR, ROUND, RUN_COUNTS, dic_traffic_env_conf, if_gui=False)

run_name = os.path.basename(os.path.normpath(MODEL_DIR))
replay_path = os.path.join("frontend", "web", "replays", run_name, "replayLogFile.txt")
size = os.path.getsize(replay_path) if os.path.exists(replay_path) else -1
print(f"\nDone. Replay: {replay_path} ({size} bytes)")
