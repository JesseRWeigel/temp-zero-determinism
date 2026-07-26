#!/usr/bin/env python3
"""Test WHY the first generation differs from every subsequent one.

The main sweep found something specific: at temperature 0, when a prompt diverges, it
diverges in exactly one way. Repetition 0 produces one output and repetitions 1 through 7
produce a different output, byte-identical to each other. That is not random
nondeterminism, which would scatter outputs across many variants. It is a systematic
difference between the first evaluation of a prompt and later ones.

The obvious suspect is prompt caching. Ollama and llama.cpp keep the KV state of a
recently evaluated prefix. The first request computes the prefix from scratch; an
identical follow-up reuses the cached state and skips that computation. Different code
paths mean a different floating point reduction order, and greedy decoding turns a tiny
logit difference into a different token whenever two candidates are nearly tied.

Three conditions distinguish caching from the alternatives:

  repeat        A, A, A, A            the baseline. Divergence expected at index 0 only.
  interleaved   A, B, A, B, A, B      each A is preceded by a different prompt. If the
                                      cache is the cause, every A after the first still
                                      differs from the first, and they agree with each
                                      other, because each is a cache hit on A's prefix
                                      only if the cache holds more than one entry. If
                                      the cache holds one entry, every A is a cold miss
                                      and they should all agree with repetition 0.
  reload        A, unload, A, unload  the model is evicted between calls, so nothing can
                                      be cached. If divergence vanishes here, caching is
                                      confirmed. If it persists, the cause is elsewhere.

    mechanism.py --model gemma4:e4b --prompts math-04,code-05 --reps 6
"""

import argparse
import json
import pathlib
import time

from harness import generate, unload

HERE = pathlib.Path(__file__).resolve().parent


def group(texts):
    """Return output-identity groups as lists of repetition indices."""
    g = {}
    for i, t in enumerate(texts):
        g.setdefault(t, []).append(i)
    return sorted(g.values(), key=len, reverse=True)


def run(model, prompt_text, other_text, reps, num_predict):
    out = {}

    # repeat: the baseline that produced the finding
    out["repeat"] = [generate(model, prompt_text, num_predict=num_predict)[0]
                     for _ in range(reps)]

    # interleaved: a different prompt between every repetition
    inter = []
    for _ in range(reps):
        generate(model, other_text, num_predict=32)
        inter.append(generate(model, prompt_text, num_predict=num_predict)[0])
    out["interleaved"] = inter

    # reload: evict the model so no cache can survive between calls
    rel = []
    for _ in range(reps):
        unload(model)
        rel.append(generate(model, prompt_text, num_predict=num_predict)[0])
    out["reload"] = rel

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--prompts", default="", help="comma separated prompt ids from prompts.json")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--num-predict", type=int, default=250)
    ap.add_argument("--out", default="raw/mechanism.json")
    a = ap.parse_args()

    catalog = {p["id"]: p["text"] for p in
               json.loads((HERE / "prompts.json").read_text())["prompts"]}
    ids = [p.strip() for p in a.prompts.split(",") if p.strip()] or list(catalog)[:2]
    other = catalog[[k for k in catalog if k not in ids][0]]

    result = {"meta": {"model": a.model, "reps": a.reps, "num_predict": a.num_predict,
                       "prompts": ids, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
              "results": {}}

    for pid in ids:
        print(f"{pid}:", flush=True)
        conds = run(a.model, catalog[pid], other, a.reps, a.num_predict)
        summary = {}
        for cond, texts in conds.items():
            groups = group(texts)
            summary[cond] = {
                "unique_outputs": len(groups),
                "groups": groups,
                "first_is_alone": groups[-1] == [0] and len(groups) == 2,
                "all_identical": len(groups) == 1,
            }
            print(f"  {cond:12s} unique={len(groups)} groups={groups}", flush=True)
        result["results"][pid] = {"summary": summary, "raw": conds}

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"\nwrote {a.out}")

    # State the reading plainly, and state the alternative if the evidence does not fit.
    reload_clean = all(r["summary"]["reload"]["all_identical"] for r in result["results"].values())
    repeat_dirty = any(not r["summary"]["repeat"]["all_identical"] for r in result["results"].values())
    print()
    if repeat_dirty and reload_clean:
        print("READING: divergence appears when a cache can persist and vanishes when the "
              "model is evicted between calls. Prompt or KV caching is the cause.")
    elif repeat_dirty:
        print("READING: divergence survives model eviction, so caching is NOT the whole "
              "story. Look at allocator state or kernel selection instead.")
    else:
        print("READING: no divergence reproduced in this run. The effect may need more "
              "repetitions or longer generations than this configuration used.")


if __name__ == "__main__":
    main()
