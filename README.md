# temp-zero-determinism

Temperature 0 is not reproducible, and the reason is more interesting than floating point.

Catalog task `EVAL-026`. 600 generations against a local model through Ollama.

## The finding

**36% of prompts produced non-identical output across 8 repetitions at temperature 0.**
Identical request, identical model, greedy decoding, and one of the eight came back
different.

The shape of that divergence is the actual result. Random floating point nondeterminism
would scatter outputs across many variants. This does not. In every diverged case there are
**exactly two distinct outputs**, and in serial requests the odd one out is **always the
first repetition**:

```
serial      9 of 25 prompts diverged, all with the pattern [[1,2,3,4,5,6,7], [0]]
concurrent  the same 9 prompts, odd index varies: 0,0,0,0,0, 1,1, 3,3
seeded      the same 9 prompts, 36.0%. An explicit seed changes nothing.
```

Susceptibility is a property of the prompt. Timing decides which repetition differs.

## Why: a cold path and a warm path, not noise

Ollama and llama.cpp cache the evaluated KV state of a prompt prefix. The first request
computes it; an identical follow-up reuses it. The two routes reduce floating point sums in
a different order, and greedy decoding turns a hair's-breadth logit difference into a
different token whenever two candidates are nearly tied. Once one token differs, the rest
of the generation follows it.

`mechanism.py` tests this directly with three sequences:

| Sequence | Unique outputs | What it means |
|---|---|---|
| `repeat` A,A,A,A,A,A | **2**, groups `[[1,2,3,4,5],[0]]` | run 0 cold, runs 1-5 warm |
| `interleaved` A,B,A,B,A,B | **1**, all identical | a different prompt between each A evicts the cache, so every A is cold |
| `reload` model unloaded between calls | **1**, all identical | nothing can be cached, so every run is cold |

This is exactly what caching predicts, and it reframes the result. Neither path is broken.
The cold path is self-consistent and the warm path is self-consistent. **Only a sequence
that mixes the two produces divergence.** Repeating a prompt is precisely such a sequence,
which is why the naive reproducibility test is the one that fails.

## What to do about it

1. **Discard the first generation** if you need a reproducible sequence, or warm the cache
   before you start measuring.
2. **Do not compare a cold run against a warm run.** An eval that runs each prompt once,
   fresh, is internally consistent. An eval that repeats prompts is not, unless it
   normalizes for cache state.
3. **A seed will not save you.** Seeding is about sampling, and at temperature 0 there is no
   sampling to control. Measured: identical 36% with and without.
4. **Short answers hide this.** Every diverged prompt was a long generation (640 to 940
   characters, all arithmetic or code). Short factual answers never diverged in 80 runs,
   because a near-tied argmax needs room to propagate before it changes anything visible.

## By prompt family

Diverged prompts, identical across all three conditions:

| Family | Diverged | Total |
|---|---:|---:|
| code | 5 | 5 |
| arithmetic | 4 | 10 |
| short_factual | 0 | 10 |

Only three of the five prompt families are in this run. `--limit 25` takes the first 25
prompts in file order, which does not reach the creative and instruction families. Those
are untested, not clean.

## Running it

```bash
./verify.sh                                    # fast subset, proves the harness works
python3 harness.py --model gemma4:e4b --reps 8 --limit 25 --num-predict 250 \
        --conditions serial,concurrent,seeded
python3 mechanism.py --prompts math-04,code-05 --reps 6
```

The full sweep takes roughly 15 minutes per condition on a contended GPU. Results append to
`raw/` after each condition, so an interruption does not discard earlier work.

## What this does not show

- **One model, one backend.** `gemma4:e4b` through Ollama. The mechanism is a property of
  prefix caching rather than of this model, so it should generalize, but that is an
  argument and not a measurement. Testing llama.cpp directly and HF `transformers` would
  make the claim much stronger and is the obvious next step.
- **25 prompts, 8 repetitions.** Enough to establish the pattern, which is unusually clean
  because all three conditions agree exactly, and not enough for a precise rate. Read 36%
  as "roughly a third of long generations" rather than as a calibrated number.
- **The divergence is measured in characters, not tokens.** The API does not return token
  ids, and inferring them would add a tokenizer whose version could itself drift.
- **`num_predict` was 250.** Longer generations plausibly diverge more often, since there
  is more opportunity for a near-tie. Not tested.

## Status

Verified 2026-07-26.

```
$ ./verify.sh
1. ollama reachable          ok
2. divergence detector       ok    identifies identical, divergent, and prefix cases
3. prompts file              ok    50 prompts across 5 families, ids unique
4. end to end run            ok    harness completed against gemma4:e4b
5. output schema             ok    documented schema and real generations
5 passed, 0 failed
VERIFY OK
```

`verify.sh` deliberately does **not** assert a particular divergence rate. Divergence is the
measurement, so baking this machine's result into the test would make the suite fail
elsewhere for the right reason. What it asserts is that the harness runs, produces the
documented schema, and that the detector actually detects divergence when handed divergent
input, which a detector that always answered "identical" would fail.

## License

MIT.
