"""Track E (Editing Weights): sparse weight surgery on a trained policy, the toy analog of
"GPT-6 Astra edited 0.016% of Qwen3.5-0.8B's weights" (Weight-Space Programming).
We take a CEM-trained policy, CORRUPT k weights, show the full weight vector, and ask the model to output <= B edits [index, new_value]
that restore performance. No code execution. recovery = (edited - corrupted) / (clean - corrupted), clipped to [0,1]."""
import json, os, re
import numpy as np
from .envs import ENVS, hid, n_params, evaluate
from .providers import chat
from .baselines import cem

GAMES = ["cartpole", "catch", "pong", "pendulum"]
K, B = 6, 10


def base_task(g, seed=0):
    path = f"configs/edit_{g}.json"
    if os.path.exists(path): return json.load(open(path))
    w, s = cem(g, iters=100 if g in ("pong", "pendulum") else 60, pop=96, elite=12, seed=seed)
    rng = np.random.default_rng(seed + 1); idx = sorted(rng.choice(len(w), K, replace=False).tolist())
    bad = np.array(w, float); bad[idx] = rng.uniform(-3, 3, K)
    t = {"game": g, "clean": w.tolist(), "corrupted": bad.tolist(), "corrupt_idx": idx,
         "clean_score": evaluate(w, g, 40, 99), "corrupted_score": evaluate(bad, g, 40, 99)}
    json.dump(t, open(path, "w")); return t


PROMPT = """A trained neural-network policy for {desc} has been corrupted: {k} of its {n} weights were overwritten with random values.
Architecture: input {o}, hidden {h} (tanh), outputs {a} (argmax). Flat order: W1 ({h}x{o} row-major), b1 ({h}), W2 ({a}x{h} row-major), b2 ({a}).
Corrupted weights (index: value):
{w}
You cannot run code. Using only reasoning about structure and magnitudes, output at most {b} edits that restore the policy.
Reply ONLY with a ```json block: [[index, new_value], ...]"""


def run(model, base, games=GAMES):
    from .track_write import DESC
    out, cost = {}, [0, 0]
    for g in games:
        t = base_task(g); env = ENVS[g]; w = np.array(t["corrupted"])
        txt, u = chat(model, [{"role": "user", "content": PROMPT.format(desc=DESC.get(g, g), k=K, n=len(w), o=env.obs_dim, h=hid(env), a=env.n_act, b=B,
                       w="\n".join(f"{i}: {v:.3f}" for i, v in enumerate(w)))}], base)
        cost[0] += u[0]; cost[1] += u[1]
        try: ed = json.loads(re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", txt, re.S)[-1])[:B]
        except Exception: ed = []
        w2 = w.copy(); n_ok = 0
        for i, v in ed:
            if isinstance(i, int) and 0 <= i < len(w2): w2[i] = float(v); n_ok += 1
        s = evaluate(w2, g, 40, 99); lo, hi = t["corrupted_score"], t["clean_score"]
        rec = float(np.clip((s - lo) / max(1e-9, hi - lo), 0, 1))
        hit = len(set(i for i, _ in ed if isinstance(i, int)) & set(t["corrupt_idx"]))
        out[g] = {"recovery": rec, "edits": n_ok, "hit_corrupted": hit, "score": s}
    out["INDEX"] = 100 * float(np.mean([out[g]["recovery"] for g in games])); out["tokens"] = cost
    return out
