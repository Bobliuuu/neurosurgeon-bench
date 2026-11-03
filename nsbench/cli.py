import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser(prog="nsbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines"); b.set_defaults(f="baselines")
    w = sub.add_parser("write", help="Track W: Writing Weights Index")
    w.add_argument("--models", required=True, help="file with one OpenRouter model id per line, or comma list")
    w.add_argument("--baselines", default="configs/baselines_v2.json"); w.add_argument("--base", default="openrouter"); w.add_argument("--rounds", type=int, default=1)
    w.add_argument("--out", default="results/write.jsonl"); w.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("edit", help="Track E: sparse weight surgery on a corrupted policy")
    e.add_argument("--models", required=True); e.add_argument("--base", default="openrouter"); e.add_argument("--out", default="results/edit.jsonl"); e.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("steer", help="Track S: neuron edit (replicate or agent-chosen)")
    s.add_argument("--target", required=True, help="HF open-weight model to edit"); s.add_argument("--trait", default="evil")
    s.add_argument("--edit", action="append", default=[], help="L,N,dose (repeatable); omit to let --agent choose")
    s.add_argument("--agent", help="model id that picks edits"); s.add_argument("--judge", default="openai/gpt-4.1-mini")
    s.add_argument("--base", default="openrouter"); s.add_argument("--out", default="results/steer.jsonl")
    a = ap.parse_args()
    if a.cmd == "baselines":
        runpy_main(); return
    if a.cmd == "write":
        from .track_write import run
        bl = json.load(open(a.baselines))
        models = open(a.models).read().split("\n") if os.path.exists(a.models) else a.models.split(",")
        from concurrent.futures import ThreadPoolExecutor
        models = [m.strip() for m in models if m.strip() and not m.startswith("#")]
        done = set()
        if os.path.exists(a.out): done = {json.loads(l)["model"] for l in open(a.out) if l.strip() and "error" not in json.loads(l)}
        def one(m):
            if m in done: return
            try: r = run(m, a.base, bl, a.rounds)
            except Exception as e: r = {"error": str(e)[:300]}
            r.update(model=m, t=time.time()); print(m, r.get("INDEX", r.get("error")), flush=True)
            open(a.out, "a").write(json.dumps(r) + "\n")
        with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, models))
    if a.cmd == "edit":
        from concurrent.futures import ThreadPoolExecutor
        from .track_edit import run as erun, base_task, GAMES
        for g in GAMES: base_task(g)
        models = open(a.models).read().split("\n") if os.path.exists(a.models) else a.models.split(",")
        models = [m.strip() for m in models if m.strip() and not m.startswith("#")]
        def one(m):
            try: r = erun(m, a.base)
            except Exception as ex: r = {"error": str(ex)[:300]}
            r.update(model=m, t=time.time()); print(m, r.get("INDEX", r.get("error")), flush=True)
            open(a.out, "a").write(json.dumps(r) + "\n")
        with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, models))
    if a.cmd == "steer":
        from .track_steer import Steerable, score, random_control, agent_edit
        st = Steerable(a.target)
        ed = [(int(x[0]), int(x[1]), float(x[2])) for x in (e.split(",") for e in a.edit)] or agent_edit(st, a.trait, a.agent, a.base)
        r = {"target": a.target, "trait": a.trait, "edits": ed, "agent": a.agent,
             "base": score(st, a.trait, [], a.judge, a.base), "edited": score(st, a.trait, ed, a.judge, a.base),
             "random_ctrl": random_control(st, a.trait, ed, a.judge, a.base)}
        print(json.dumps(r, indent=1)); open(a.out, "a").write(json.dumps(r) + "\n")


def runpy_main():
    import runpy; runpy.run_module("nsbench.baselines", run_name="__main__")

if __name__ == "__main__":
    main()
