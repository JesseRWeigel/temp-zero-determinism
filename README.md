# temp-zero-determinism

Temperature 0 is not reproducible, and the two backends tested fail for different reasons.

> Measurements described here were taken on one development machine: an RTX 5090 with
> 32 GB of VRAM, 12 cores, 48 GB of RAM, running Linux under WSL2. Numbers from your own
> hardware will differ.

Catalog task `EVAL-026`. 1,200 generations across two backends: a local model through
Ollama and a hosted model through the Gemini API.

**[Read this on the web](https://jesserweigel.github.io/temp-zero-determinism/)**

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

## Two backends, two different mechanisms

Adding a hosted backend changed the conclusion, and the change is the interesting part.

| | Ollama, `gemma4:e4b` | Gemini, `gemini-3-flash-preview` |
|---|---|---|
| Divergence rate | 36% | **64%** |
| Unique outputs per diverged prompt | Always exactly **2** | **2, 3, 4, and 5** |
| Is the odd run always the first? | Serial 9/9, seeded 9/9 | **0 of 16, in every condition** |
| Rate with an explicit seed | 36%, unchanged | 64%, unchanged |

**Ollama's divergence is a two-state split and it is predictable.** Every diverged prompt
produced exactly two distinct outputs, and under serial requests the odd one out was the first
repetition every single time. `mechanism.py` confirms why: repeating a prompt gives two outputs,
interleaving a different prompt between repetitions gives one, and unloading the model between
calls gives one. That is prefix KV caching. The cold path and the warm path are each internally
consistent, and only a sequence mixing them diverges.

**Gemini's divergence is not that, at all.** Eight byte-identical requests produced up to five
distinct outputs, and the first-repetition signature is entirely absent: 0 of 16 diverged prompts
in any condition. There is no two-state structure to find. This is what continuous
nondeterminism looks like, and it is consistent with continuous batching on a shared server,
where a request is batched with whatever else happens to be in flight, so the reduction order
differs on every call rather than between two states.

So the caching explanation is correct for the local backend and **does not generalise**. An
earlier version of this README said "only a sequence that mixes the two produces divergence".
That is true of Ollama and false of Gemini, where repeated identical requests scatter. The
honest conclusion is stronger than the original one:

> Temperature 0 is not reproducible on either backend. The local one fails in a predictable,
> explainable way that you can work around. The hosted one fails continuously, at nearly twice
> the rate, and there is nothing on the client side to work around it with.

An explicit seed changes nothing on either, which makes sense: seeding controls sampling, and
greedy decoding has no sampling to control.

## What to do about it

1. **Do not assume a hosted API is more deterministic than a local one.** It was worse here, 64%
   against 36%, and unlike the local case there was no pattern to exploit.
2. **On a local backend, discard the first generation** or warm the cache before measuring. That
   removes the entire effect, because the effect is one cold run among warm ones.
3. **On a hosted backend, budget for it.** If an eval needs a stable answer, run the prompt
   several times and take a majority, or accept that a single call is a sample rather than a
   measurement.
4. **Never compare a cold run against a warm run** on a local backend. An eval that runs each
   prompt once, fresh, is internally consistent. One that repeats prompts is not.
5. **A seed will not save you.** Measured identical with and without, on both backends.
6. **Short answers hide all of this.** Every diverged prompt was a long generation. Short
   factual answers never diverged in 80 local runs, because a near-tied argmax needs room to
   propagate before it changes anything visible.

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
- **25 prompts, 8 repetitions per condition.** Enough to establish the patterns, which are
  unusually clean because all three conditions agree exactly on both backends, and not enough
  for precise rates. Read 36% and 64% as "about a third" and "about two thirds" of long
  generations rather than as calibrated numbers.
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
measurement, so baking the development machine's result into the test would make the suite fail
elsewhere for the right reason. What it asserts is that the harness runs, produces the
documented schema, and that the detector actually detects divergence when handed divergent
input, which a detector that always answered "identical" would fail.

## License

MIT.
