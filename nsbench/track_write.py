"""Track W (Writing Weights Index): LLM writes MLP weights from scratch, zero gradients, one-shot or with feedback rounds.
Index = mean over games of 100*q/(1+q), q = max(0, score-random)/(ref-random).  q=1 (matches ref) -> 50."""
import json, re
import numpy as np
from .envs import ENVS, HIDDEN, n_params, evaluate
from .providers import chat

PROMPT = """You are writing the weights of a tiny neural network BY HAND. No training, no code execution.
Game: {desc}
Observation dim = {o}. Actions = {a} (policy = argmax of outputs).
Network: h = tanh(W1 @ obs + b1); logits = W2 @ h + b2. Hidden size {h}.
Flat parameter order: W1 (row-major {h}x{o}), b1 ({h}), W2 (row-major {a}x{h}), b2 ({a}). Total {n} numbers.
Reason about the dynamics, then output ONLY a final JSON list of exactly {n} floats in a ```json block."""
DESC = {
    "cartpole": "CartPole. obs=[cart_x, cart_vel, pole_angle, pole_ang_vel]. Action 1 pushes right, 0 left. Reward +1/step until pole falls (max 500).",
    "mountaincar": "MountainCar. obs=[position, velocity]; goal at position 0.5; valley at about -0.5. Actions 0=push left,1=none,2=push right. Reward -1/step (max 200 steps).",
    "catch": "Catch. obs=[ball_x, ball_y, paddle_x] each in [0,1]. Actions 0=paddle left,1=stay,2=right. +1 per ball caught, -1 per miss, 10 balls.",
}


def parse(text, n):
    m = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S) or re.findall(r"(\[[^\[\]]*\])", text, re.S)
    for blob in reversed(m):
        try:
            w = json.loads(blob)
            if len(w) == n and all(isinstance(x, (int, float)) for x in w): return w
        except Exception: pass
    return None


def run(model, base, baselines, rounds=1, episodes=30):
    res, cost = {}, [0, 0]
    for g, env_cls in ENVS.items():
        n = n_params(env_cls)
        msgs = [{"role": "user", "content": PROMPT.format(desc=DESC[g], o=env_cls.obs_dim, a=env_cls.n_act, h=HIDDEN, n=n)}]
        best = None
        for r in range(rounds):
            txt, u = chat(model, msgs, base); cost[0] += u[0]; cost[1] += u[1]
            w = parse(txt, n)
            if w is None:
                msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": f"Invalid: need a JSON list of exactly {n} numbers."}]
                continue
            s = evaluate(w, g, episodes, 4242)
            best = s if best is None else max(best, s)
            msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": f"Your weights scored {s:.2f} (random={baselines[g]['random']:.2f}, reference={baselines[g]['ref']:.2f}). Revise and output a new full JSON list."}]
        b = baselines[g]; score = best if best is not None else b["random"]
        q = max(0.0, score - b["random"]) / max(1e-9, b["ref"] - b["random"])
        res[g] = {"score": score, "q": q, "index": 100 * q / (1 + q)}
    res["INDEX"] = float(np.mean([res[g]["index"] for g in ENVS]))
    res["tokens"] = cost
    return res
