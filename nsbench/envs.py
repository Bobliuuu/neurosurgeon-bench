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


class Pong:
    """Paddle on left edge (y in [0,1], speed 0.025). Ball bounces off top/bottom/right. obs=[bx,by,vx,vy,py]. 10 serves; +1 return, -1 miss.
    Ball is faster than the paddle in y, so the policy must anticipate, not just track."""
    name, obs_dim, n_act, max_steps, hidden = "pong", 5, 3, 1500, 12

    def _serve(self):
        self.bx, self.by = 0.9, float(self.rng.uniform(0.1, 0.9))
        self.vx = -float(self.rng.uniform(0.025, 0.04)); self.vy = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.01, 0.04))

    def reset(self, rng):
        self.rng = rng; self.py = 0.5; self.n = 0; self.t = 0; self._serve(); return self._obs()

    def _obs(self): return np.array([self.bx, self.by, self.vx * 20, self.vy * 20, self.py])

    def step(self, a):
        self.py = float(np.clip(self.py + (a - 1) * 0.025, 0, 1)); self.t += 1; r = 0.0
        self.bx += self.vx; self.by += self.vy
        if self.by < 0 or self.by > 1: self.vy = -self.vy; self.by = float(np.clip(self.by, 0, 1))
        if self.bx > 1: self.vx = -self.vx; self.bx = 1.0
        if self.bx <= 0:
            r = 1.0 if abs(self.by - self.py) < 0.08 else -1.0; self.n += 1; self._serve()
        return self._obs(), r, self.n >= 10 or self.t >= self.max_steps


class Pendulum:
    """Underpowered pendulum swing-up (torque in {-1,0,+1}*1.0, gravity forces pumping). obs=[cos,sin,thdot/8]. 200 steps, reward -(th^2+0.1*thd^2)."""
    name, obs_dim, n_act, max_steps, hidden = "pendulum", 3, 3, 200, 12

    def reset(self, rng):
        self.th = float(rng.uniform(-math.pi, math.pi)); self.thd = float(rng.uniform(-1, 1)); self.t = 0; return self._obs()

    def _obs(self): return np.array([math.cos(self.th), math.sin(self.th), self.thd / 8])

    def step(self, a):
        u = (a - 1) * 1.0; th = ((self.th + math.pi) % (2 * math.pi)) - math.pi
        r = -(th ** 2 + 0.1 * self.thd ** 2)
        self.thd = float(np.clip(self.thd + (15 * math.sin(self.th) + 3 * u) * 0.05, -8, 8)); self.th += self.thd * 0.05; self.t += 1
        return self._obs(), r, self.t >= self.max_steps


ENVS = {c.name: c for c in (CartPole, MountainCar, Catch, Pong, Pendulum)}
HIDDEN = 8


def hid(env): return getattr(env, "hidden", HIDDEN)


def n_params(env):
    h = hid(env)
    return h * env.obs_dim + h + env.n_act * h + env.n_act


def unpack(w, env):
    o, h, a = env.obs_dim, hid(env), env.n_act
    w = np.asarray(w, float); i = 0
    W1 = w[i:i + h * o].reshape(h, o); i += h * o
    b1 = w[i:i + h]; i += h
    W2 = w[i:i + a * h].reshape(a, h); i += a * h
    return W1, b1, W2, w[i:i + a]


def evaluate(w, env_name, episodes=20, seed=0):
    if env_name in CIRCUITS: return evaluate_circuit(w, env_name)
    env = ENVS[env_name](); rng = np.random.default_rng(seed)
    W1, b1, W2, b2 = unpack(w, env); tot = 0.0
    for _ in range(episodes):
        o, done = env.reset(rng), False
        while not done:
            a = int(np.argmax(W2 @ np.tanh(W1 @ o + b1) + b2))
            o, r, done = env.step(a); tot += r
    return tot / episodes


# ---- circuit tasks: no env, exact input->output mapping; outputs thresholded at 0, score = bit accuracy ----
import itertools


class Circuit:
    kind = "circuit"; hidden = 12


class Parity8(Circuit):
    name, obs_dim, n_act, chance, hidden = "parity8", 8, 1, 0.5, 8
    X = np.array(list(itertools.product([0, 1], repeat=8)), float); Y = (X.sum(1) % 2)[:, None]


def _add_bits(X):
    s = (X[:, :3] @ [4, 2, 1] + X[:, 3:] @ [4, 2, 1]).astype(int)
    return np.stack([(s >> k) & 1 for k in (3, 2, 1, 0)], 1).astype(float)


class Add3(Circuit):
    name, obs_dim, n_act, chance, hidden = "add3", 6, 4, 0.5, 16
    X = np.array(list(itertools.product([0, 1], repeat=6)), float)
    Y = _add_bits(X)


class Sort3(Circuit):
    """3 reals in [0,1] -> output 3 bits: is a>b, b>c, a>c (comparison network)"""
    name, obs_dim, n_act, chance, hidden = "cmp3", 3, 3, 0.5, 12
    X = np.random.default_rng(0).random((512, 3)); Y = np.stack([X[:, 0] > X[:, 1], X[:, 1] > X[:, 2], X[:, 0] > X[:, 2]], 1).astype(float)


CIRCUITS = {c.name: c for c in (Parity8, Add3, Sort3)}


def evaluate_circuit(w, name):
    c = CIRCUITS[name]; W1, b1, W2, b2 = unpack(w, c)
    out = np.tanh(c.X @ W1.T + b1) @ W2.T + b2
    return float(((out > 0) == (c.Y > 0.5)).mean())
