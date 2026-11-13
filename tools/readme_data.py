#!/usr/bin/env python3
"""results/*.jsonl + configs/baselines*.json -> results/derived/readme_data.json (README / video numbers)."""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def rows(path):
    out = []
    for l in open(os.path.join(ROOT, path)):
        l = l.strip()
        if l:
            d = json.loads(l)
            if "error" not in d and "INDEX" in d:
                out.append(d)
    return out


d = {}
# Track W round 1: three games, one-shot hand-written weights
r1 = {}
for r in rows("results/write_r1.jsonl"):
    r1[r["model"]] = r  # keep the latest row per model
r1 = list(r1.values())
r1.sort(key=lambda r: -r["INDEX"])
d["w1_models"] = len(r1)
d["w1_top"] = r1[0]["INDEX"]
d["w1_ge50"] = sum(1 for r in r1 if r["INDEX"] >= 50)
d["w1_zero"] = sum(1 for r in r1 if r["INDEX"] == 0)
d["w1_gt0"] = sum(1 for r in r1 if r["INDEX"] > 0)
d["w1_median"] = sorted(r["INDEX"] for r in r1)[len(r1) // 2]
d["w1_hist"] = [{"band": b, "models": sum(1 for r in r1 if lo <= r["INDEX"] < hi)}
                for b, lo, hi in [("0", 0, 1e-9), ("0-20", 1e-9, 20), ("20-40", 20, 40), ("40-50", 40, 50), ("50+", 50, 101)]]
d["w1_top10"] = [{"model": r["model"].split("/")[-1], "index": r["INDEX"]} for r in r1[:10]]
d["w1_table"] = [{"model": r["model"], "cartpole": r["cartpole"]["index"], "mountaincar": r["mountaincar"]["index"],
                  "catch": r["catch"]["index"], "index": r["INDEX"]} for r in r1[:12]]
# per-game: fraction of models reaching >= 40 index
for g in ["cartpole", "mountaincar", "catch"]:
    d[f"w1_{g}_ge40"] = sum(1 for r in r1 if r[g]["index"] >= 40)

# Track W v2: harder tasks, pilot of 6 models
tasks = ["cartpole", "mountaincar", "catch", "pong", "pendulum", "parity8", "add3", "cmp3"]
v2 = rows("results/pilot_write_v2.jsonl")
d["w2_models"] = len(v2)
d["w2_tasks"] = len(tasks)
d["w2_rows"] = [{"model": r["model"], "index": r["INDEX"], **{t: r[t]["index"] for t in tasks}} for r in sorted(v2, key=lambda r: -r["INDEX"])]
d["w2_top"] = max(r["INDEX"] for r in v2)
d["w2_long"] = [{"task": t, "model": r["model"].split("/")[-1], "index": r[t]["index"]}
                for t in ["pong", "pendulum", "parity8"] for r in v2 if r["INDEX"] > 40]
d["w2_hard"] = [{"task": t, "model": r["model"].split("/")[-1], "index": r[t]["index"]} for t in ["pong", "pendulum", "parity8"] for r in v2]
# Track E: edit a corrupted policy, 6 models
ed = rows("results/pilot_edit.jsonl")
games = ["cartpole", "catch", "pong", "pendulum"]
d["e_models"] = len(ed)
d["e_top"] = max(r["INDEX"] for r in ed)
d["e_rows"] = [{"model": r["model"], "index": r["INDEX"], **{g: r[g]["recovery"] * 100 for g in games}} for r in sorted(ed, key=lambda r: -r["INDEX"])]
d["e_chart"] = [{"game": g, "model": r["model"].split("/")[-1], "recovery": r[g]["recovery"] * 100} for g in games for r in ed if r["INDEX"] > 10]
d["e_full"] = sum(1 for r in ed for g in games if r[g]["recovery"] >= 0.999)
# baselines
b = json.load(open(os.path.join(ROOT, "configs/baselines_v2.json")))
d["baselines"] = [{"task": k, "random": v["random"], "ref": v["ref"]} for k, v in b.items()]
os.makedirs(os.path.join(ROOT, "results", "derived"), exist_ok=True)
json.dump(d, open(os.path.join(ROOT, "results", "derived", "readme_data.json"), "w"), indent=1)
print("ok", {k: v for k, v in d.items() if not isinstance(v, list)})
