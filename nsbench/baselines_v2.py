"""v2 baselines: random floor + CEM ceiling for control games; circuits use ref = halfway between chance and perfect (so perfect -> q=2 -> index 66.7)."""
import json, sys
import numpy as np
from multiprocessing import Pool
from .envs import ENVS, CIRCUITS, n_params, evaluate
from .baselines import random_score, cem, mountaincar_hand

CONTROL = ["cartpole", "mountaincar", "catch", "pong", "pendulum"]


def one(n):
    if n in CIRCUITS:
        c = CIRCUITS[n]; return n, {"random": c.chance, "ref": c.chance + (1 - c.chance) / 2}
    iters = {"pong": 120, "pendulum": 100}.get(n, 60)
    w, s = mountaincar_hand() if n == "mountaincar" else cem(n, iters=iters, pop=96, elite=12)
    r = {"random": random_score(n, 60), "ref": s}; print(n, r, file=sys.stderr, flush=True); return n, r


if __name__ == "__main__":
    with Pool(8) as p: out = dict(p.map(one, CONTROL + list(CIRCUITS)))
    json.dump(out, open("configs/baselines_v2.json", "w"), indent=1)
