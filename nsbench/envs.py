"""Tiny dependency-free control tasks. Each env: reset(rng)->obs, step(a)->(obs,r,done)."""
import math
import numpy as np


class CartPole:
    name, obs_dim, n_act, max_steps = "cartpole", 4, 2, 500

    def reset(self, rng):
        self.s = rng.uniform(-0.05, 0.05, 4); self.t = 0
        return self.s.copy()

    def step(self, a):
        x, xd, th, thd = self.s
        f = 10.0 if a == 1 else -10.0
        c, s = math.cos(th), math.sin(th)
        tmp = (f + 0.05 * thd ** 2 * s) / 1.1
        tha = (9.8 * s - c * tmp) / (0.5 * (4 / 3 - 0.1 * c ** 2 / 1.1))
        xa = tmp - 0.05 * tha * c / 1.1
        self.s = np.array([x + .02 * xd, xd + .02 * xa, th + .02 * thd, thd + .02 * tha])
        self.t += 1
        done = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 0.2095 or self.t >= self.max_steps
        return self.s.copy(), 1.0, done


class MountainCar:
    name, obs_dim, n_act, max_steps = "mountaincar", 2, 3, 200

    def reset(self, rng):
        self.s = np.array([rng.uniform(-0.6, -0.4), 0.0]); self.t = 0
        return self.s.copy()

    def step(self, a):
        p, v = self.s
        v += (a - 1) * 0.001 + math.cos(3 * p) * -0.0025
        v = float(np.clip(v, -0.07, 0.07)); p = float(np.clip(p + v, -1.2, 0.6))
        if p == -1.2 and v < 0: v = 0.0
        self.s = np.array([p, v]); self.t += 1
        return self.s.copy(), -1.0, p >= 0.5 or self.t >= self.max_steps


class Catch:
    """5x5 grid; ball falls, paddle on bottom row. obs=[ball_x, ball_y, paddle_x] in [0,1]. 10 balls/episode."""
    name, obs_dim, n_act, max_steps = "catch", 3, 3, 40

    def _ball(self): self.bx, self.by = int(self.rng.integers(5)), 0

    def reset(self, rng):
        self.rng = rng; self.px = 2; self.balls = 0; self.t = 0; self._ball()
        return self._obs()

    def _obs(self): return np.array([self.bx / 4, self.by / 4, self.px / 4])

    def step(self, a):
        self.px = int(np.clip(self.px + (a - 1), 0, 4)); self.by += 1; self.t += 1; r = 0.0
        if self.by == 4:
            r = 1.0 if self.px == self.bx else -1.0; self.balls += 1; self._ball()
        return self._obs(), r, self.balls >= 10 or self.t >= self.max_steps


ENVS = {c.name: c for c in (CartPole, MountainCar, Catch)}
HIDDEN = 8


def n_params(env):
    return HIDDEN * env.obs_dim + HIDDEN + env.n_act * HIDDEN + env.n_act


def unpack(w, env):
    o, h, a = env.obs_dim, HIDDEN, env.n_act
    w = np.asarray(w, float); i = 0
    W1 = w[i:i + h * o].reshape(h, o); i += h * o
    b1 = w[i:i + h]; i += h
    W2 = w[i:i + a * h].reshape(a, h); i += a * h
    return W1, b1, W2, w[i:i + a]


def evaluate(w, env_name, episodes=20, seed=0):
    env = ENVS[env_name](); rng = np.random.default_rng(seed)
    W1, b1, W2, b2 = unpack(w, env); tot = 0.0
    for _ in range(episodes):
        o, done = env.reset(rng), False
        while not done:
            a = int(np.argmax(W2 @ np.tanh(W1 @ o + b1) + b2))
            o, r, done = env.step(a); tot += r
    return tot / episodes
