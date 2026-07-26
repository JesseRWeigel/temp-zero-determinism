#!/usr/bin/env bash
# Proves the harness works end to end without running the full sweep.
#
# It deliberately does NOT assert that output is deterministic or that it is not.
# Divergence is the measurement, so asserting a particular rate would bake this run's
# result into the test and the suite would fail on a different machine for the right
# reason. What is asserted is that the harness runs, produces the documented schema,
# and that its divergence detector actually detects divergence when given divergent
# input.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pass=0; fail=0
ok()  { printf '  ok    %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "1. ollama reachable"
if curl -sf -m 10 http://localhost:11434/api/tags >/dev/null; then
  ok "ollama responding on 11434"
else
  bad "ollama not reachable, cannot run"; echo "VERIFY FAILED"; exit 1
fi

echo
echo "2. divergence detector unit checks"
python3 - <<'PY' && ok "detector identifies identical, divergent, and prefix cases" || bad "detector unit checks"
import sys
from harness import first_divergence, similarity, analyze_condition

assert first_divergence("abc", "abc") is None, "identical must report None"
assert first_divergence("abc", "abd") == 2, "should find index 2"
assert first_divergence("abc", "abcdef") == 3, "prefix case should report the split"
assert similarity("abc", "abc") == 1.0

# A condition where every run is identical must report zero divergence.
same = {"p1": [{"text": "hello", "eval_count": 1, "error": None} for _ in range(4)]}
r = analyze_condition(same)
assert r["prompts_with_divergence"] == 0, r
assert r["per_prompt"][0]["identical"] is True

# And one with a real difference must report it. This is the check that matters:
# a detector that always says "identical" would pass the case above.
diff = {"p1": [{"text": t, "eval_count": 1, "error": None}
               for t in ["hello world", "hello world", "hello worlx", "hello world"]]}
r = analyze_condition(diff)
assert r["prompts_with_divergence"] == 1, r
assert r["per_prompt"][0]["n_diverged"] == 1, r
assert r["per_prompt"][0]["min_first_divergence"] == 10, r
assert r["per_prompt"][0]["unique_outputs"] == 2, r
print("detector checks passed", file=sys.stderr)
PY

echo
echo "3. prompts file is well formed"
python3 - <<'PY' && ok "50 prompts across 5 families, ids unique" || bad "prompts.json malformed"
import json, collections
d = json.load(open("prompts.json"))
ps = d["prompts"]
assert len(ps) == 50, f"expected 50 prompts, found {len(ps)}"
assert len({p["id"] for p in ps}) == 50, "duplicate prompt ids"
fams = collections.Counter(p["family"] for p in ps)
assert len(fams) == 5, f"expected 5 families, found {dict(fams)}"
assert all(p["text"].strip() for p in ps), "empty prompt text"
PY

echo
echo "4. end to end run against a real model"
MODEL="${VERIFY_MODEL:-gemma4:e4b}"
if timeout 600 python3 harness.py --model "$MODEL" --quick --out "$TMP/out.json" >"$TMP/run.log" 2>&1; then
  ok "harness completed against $MODEL"
else
  bad "harness run failed"; tail -12 "$TMP/run.log" | sed 's/^/        /'
fi

echo
echo "5. output schema"
python3 - "$TMP/out.json" <<'PY' && ok "output has the documented schema and real generations" || bad "output schema"
import json, sys
d = json.load(open(sys.argv[1]))
assert "meta" in d and "conditions" in d and "raw" in d
for key in ("model", "reps", "prompts", "conditions", "generated_at"):
    assert key in d["meta"], f"meta missing {key}"
for cond in ("serial", "concurrent"):
    c = d["conditions"][cond]
    for key in ("prompts", "prompts_with_divergence", "divergence_rate_pct", "per_prompt"):
        assert key in c, f"{cond} missing {key}"
    assert c["prompts"] == 3, c["prompts"]
    # every recorded generation must be real text, not an empty placeholder
    texts = [r["text"] for runs in d["raw"][cond].values() for r in runs]
    assert len(texts) == 12, f"{cond}: expected 12 generations, got {len(texts)}"
    assert all(t.strip() for t in texts), f"{cond}: an empty generation was recorded"
PY

echo
printf '%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || { echo "VERIFY FAILED"; exit 1; }
echo "VERIFY OK"
