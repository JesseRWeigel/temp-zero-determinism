#!/usr/bin/env python3
"""Measure whether temperature 0 actually produces identical output.

Greedy decoding is arithmetically deterministic on paper. In practice a server can
return different text for byte-identical requests, because floating point addition is
not associative and the reduction order inside a GPU kernel depends on how work was
tiled. Batch size changes the tiling. A server that batches concurrent requests
therefore has a nondeterminism source that a serial client never sees.

This harness varies the conditions that plausibly matter and measures divergence:

  serial      requests one at a time, the control
  concurrent  N requests in flight at once, which is where batching effects appear
  seeded      an explicit seed, to test whether it buys anything at temperature 0
  reload      the model unloaded between repetitions, testing allocation-order effects

    harness.py --model gemma4:e4b --reps 10 --out raw/gemma4.json
    harness.py --quick                     # tiny configuration used by verify.sh

Writes one JSON file per model containing every generation, so analysis can be redone
without re-running the sweep.
"""

import argparse
import difflib
import json
import pathlib
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://localhost:11434"
HERE = pathlib.Path(__file__).resolve().parent


def generate(model, prompt, seed=None, num_predict=200, keep_alive="5m"):
    """One completion at temperature 0. Returns (text, eval_count, error)."""
    opts = {"temperature": 0.0, "num_predict": num_predict}
    if seed is not None:
        opts["seed"] = seed
    payload = {"model": model, "prompt": prompt, "stream": False,
               "think": False, "options": opts, "keep_alive": keep_alive}
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
        return d.get("response", ""), d.get("eval_count"), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return "", None, str(e)[:200]


def unload(model):
    """keep_alive 0 evicts the model, so the next call reallocates from scratch."""
    generate(model, "x", num_predict=1, keep_alive="0")
    time.sleep(1.5)


def first_divergence(a, b):
    """Index of the first differing character, or None if identical.

    Character index rather than token index: the API does not return token ids, and
    inferring them would add a tokenizer dependency whose own version could drift.
    The character position is reported honestly as what it is.
    """
    if a == b:
        return None
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def similarity(a, b):
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


def run_condition(model, prompts, reps, condition, seed, num_predict, workers):
    """Return {prompt_id: [text, ...]} with `reps` generations per prompt."""
    out = {}
    for p in prompts:
        pid, text = p["id"], p["text"]
        if condition == "concurrent":
            # All repetitions in flight together. If the server batches them, they
            # share a kernel launch and the reduction order can differ from serial.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(
                    lambda _: generate(model, text, seed, num_predict),
                    range(reps)))
        else:
            res = []
            for _ in range(reps):
                if condition == "reload":
                    unload(model)
                res.append(generate(model, text, seed, num_predict))
        out[pid] = [{"text": t, "eval_count": n, "error": e} for t, n, e in res]
    return out


def analyze_condition(gens):
    """Divergence stats for one condition, comparing every repetition to the first."""
    rows = []
    for pid, runs in gens.items():
        texts = [r["text"] for r in runs if not r["error"]]
        errs = sum(1 for r in runs if r["error"])
        if len(texts) < 2:
            rows.append({"prompt": pid, "n": len(texts), "errors": errs,
                         "identical": None, "note": "too few successful runs"})
            continue
        base = texts[0]
        divs = [first_divergence(base, t) for t in texts[1:]]
        differing = [d for d in divs if d is not None]
        rows.append({
            "prompt": pid,
            "n": len(texts),
            "errors": errs,
            "unique_outputs": len(set(texts)),
            "identical": len(differing) == 0,
            "n_diverged": len(differing),
            "first_divergence_chars": differing,
            "min_first_divergence": min(differing) if differing else None,
            "mean_similarity": round(
                sum(similarity(base, t) for t in texts[1:]) / max(len(texts) - 1, 1), 4),
            "base_len": len(base),
        })
    total = len(rows)
    diverged = sum(1 for r in rows if r.get("identical") is False)
    return {
        "prompts": total,
        "prompts_with_divergence": diverged,
        "divergence_rate_pct": round(100 * diverged / total, 1) if total else None,
        "per_prompt": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--num-predict", type=int, default=200)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, help="use only the first N prompts")
    ap.add_argument("--conditions", default="serial,concurrent,seeded,reload")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="3 prompts, 4 reps, serial and concurrent only")
    a = ap.parse_args()

    prompts = json.loads((HERE / "prompts.json").read_text())["prompts"]
    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    reps, num_predict = a.reps, a.num_predict

    if a.quick:
        prompts, reps, num_predict = prompts[:3], 4, 80
        conditions = ["serial", "concurrent"]
    elif a.limit:
        prompts = prompts[:a.limit]

    out_path = pathlib.Path(a.out or f"raw/{a.model.replace(':', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {"meta": {
        "model": a.model, "reps": reps, "prompts": len(prompts),
        "num_predict": num_predict, "conditions": conditions,
        "workers": a.workers, "quick": a.quick,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, "conditions": {}}

    for cond in conditions:
        seed = 42 if cond == "seeded" else None
        t0 = time.time()
        print(f"  {cond}: {len(prompts)} prompts x {reps} reps", flush=True)
        gens = run_condition(a.model, prompts, reps, cond, seed, num_predict, a.workers)
        summary = analyze_condition(gens)
        summary["seconds"] = round(time.time() - t0, 1)
        summary["seed"] = seed
        result["conditions"][cond] = summary
        result.setdefault("raw", {})[cond] = gens
        print(f"    {summary['prompts_with_divergence']}/{summary['prompts']} prompts "
              f"diverged ({summary['divergence_rate_pct']}%) in {summary['seconds']}s",
              flush=True)
        # Write after every condition, not only at the end. A sweep against local models
        # runs for hours, and losing all of it because the last condition was interrupted
        # is an avoidable way to waste a GPU-day.
        result["meta"]["conditions_completed"] = list(result["conditions"])
        out_path.write_text(json.dumps(result, indent=1))

    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path}")

    # A nonzero exit here would be wrong. Divergence is the finding, not a failure.
    for cond, s in result["conditions"].items():
        print(f"{cond:11s} {s['divergence_rate_pct']:5.1f}% of prompts diverged")


if __name__ == "__main__":
    main()
