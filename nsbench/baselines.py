"""Baselines: uniform-random weights (floor) and CEM-trained policy (ceiling, stands in for PPO)."""
import json, sys
import numpy as np
from .envs import ENVS, n_params, evaluate


def random_score(env_name, n=200, seed=1):
    env = ENVS[env_name](); rng = np.random.default_rng(seed)
    return float(np.mean([evaluate(rng.uniform(-1, 1, n_params(env)), env_name, 5, seed + i) for i in range(n)]))


def mountaincar_hand():
    """CEM fails on MountainCar (sparse reward); use hand-built 'push along velocity' net as reference."""
    import numpy as np
    env = ENVS["mountaincar"]; w = np.zeros(n_params(env)); w[1] = 50.0; o = 8 * 2 + 8; w[o + 0] = -1.0; w[o + 16] = 1.0
    return w, evaluate(w, "mountaincar", 50, 777)


def cem(env_name, iters=60, pop=64, elite=8, seed=0):
    env = ENVS[env_name](); rng = np.random.default_rng(seed); d = n_params(env)
    mu, sd = np.zeros(d), np.ones(d) * 1.0; best = (-1e9, mu)
    for it in range(iters):
        P = mu + sd * rng.standard_normal((pop, d))
        S = np.array([evaluate(p, env_name, 5, 1000 + it) for p in P])
        idx = np.argsort(S)[-elite:]; mu, sd = P[idx].mean(0), P[idx].std(0) + 0.05
        if S[idx[-1]] > best[0]: best = (S[idx[-1]], P[idx[-1]])
    return best[1], evaluate(best[1], env_name, 50, 777)


if __name__ == "__main__":
    out = {}
    for n in ENVS:
        w, s = mountaincar_hand() if n == 'mountaincar' else cem(n); out[n] = {"random": random_score(n), "ref": s, "ref_weights": w.tolist()}
        print(n, out[n]["random"], out[n]["ref"], file=sys.stderr)
    json.dump(out, open("configs/baselines.json", "w"), indent=1)
