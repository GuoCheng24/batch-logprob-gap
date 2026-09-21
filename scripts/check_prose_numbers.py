"""The numbers the README states in prose must come from a committed result file.

The tables are safe: `build_tables.py` regenerates them and CI diffs the result. The prose is not.
One sentence -- "enabling vLLM's batch-invariant kernels does not shrink it: 8.0% to 7.1%, while
changing the trainer's precision goes to 3.7% in one step" -- stood for days with no file behind it
in this repository; the measurement existed, but only on the machine that ran it. An audit found it,
not a test. This guard re-derives each such figure from `results/` so the next one cannot survive.

It compares numeric tokens, which is a blind spot worth naming: a quantity spelled in words walks
straight past it. "an fp32 head removes two thirds to four fifths of the out-of-band tokens" stood
on the page while the measurement was 64.4% to 81.2% -- outside that range at both ends, and invisible
to a checker that only looks at digits. Spelled-out fractions are now refused outright.

    python scripts/check_prose_numbers.py
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _load(name):
    with open(RESULTS / name) as fh:
        return json.load(fh)


def pct(arm, data):
    return 100 * data["arms"][arm]["out"] / data["n_tokens"]


def hf_rate(model, key="bf16|b1_vs_b8"):
    r = _load(f"hfonly_{model}.json")["results"][key]
    return 100 * r["out"] / r["n"]


def flips(model):
    d = _load(f"grpo_{model}.json")
    return 100 * d["flips"] / d["n"]


def head_removal_range():
    """How much of the bf16 out-of-band rate an fp32 lm_head removes, over the Qwen-family models.

    pythia is excluded on purpose: the sentence this bounds is about the Qwen side, where the
    divergence enters early and the head can still undo it. On pythia the same arm removes 5.9%.
    """
    fractions = []
    for f in sorted(RESULTS.glob("selective_*_x4.json")):
        d = json.loads(f.read_text())
        if "pythia" in d["model"].lower():
            continue
        arms = {a["name"]: a["oob"] for a in d["arms"]}
        fractions.append(
            100 * (1 - arms["head only (ScaleRL)"] / arms["baseline bf16"])
        )
    return min(fractions), max(fractions)


def main() -> int:
    kern = _load("kernel_arms.json")
    dtype = _load("dtype_pythia-410m.json")["results"]
    pack = _load("pack_pythia-410m.json")["results"]

    def rate(d, k):
        return 100 * d[k]["out"] / d[k]["n"]

    # (what the README says, recomputed value, the figure as written)
    checks = [
        ("batch-invariant kernels, before", pct("BI0 vs bf16 b8", kern), "8.0"),
        ("batch-invariant kernels, after", pct("BI1 vs bf16 b8", kern), "7.1"),
        ("trainer precision in one step", pct("BI0 vs fp32 b8", kern), "3.7"),
        ("packing with a block-diagonal mask", rate(dtype, "bf16|pack"), "32.23"),
        ("padding, the same model and run", rate(dtype, "bf16|batch"), "31.84"),
        ("the first packing run, no mask (sd)", pack["packed x1 vs x8"]["sd"], "2.80"),
        (
            "bf16 rate range, all models",
            hf_rate("pythia-410m"),
            None,
        ),  # covered by refresh_readme.py
        ("clip-status flips, smallest", flips("Qwen2.5-0.5B-Instruct"), "1.7"),
        ("clip-status flips, largest", flips("pythia-410m"), "10.1"),
    ]
    readme = (ROOT / "README.md").read_text()
    failures = []
    for label, got, written in checks:
        if written is None:
            print(f"  {label:<38} {got:>6.2f}%  (covered by refresh_readme.py)")
            continue
        places = len(written.split(".")[1]) if "." in written else 0
        got_r = f"{round(got, places):.{places}f}"
        on_page = (
            re.search(r"(?<![\d.])" + re.escape(written) + r"(?![\d])", readme)
            is not None
        )
        ok = got_r == written and on_page
        state = "ok" if ok else ("MISMATCH" if got_r != written else "NOT ON PAGE")
        print(f"  {label:<38} page {written:>6}  results {got_r:>6}  {state}")
        if not ok:
            failures.append(
                f"{label}: page {written}, results {got_r}, on page {on_page}"
            )
    # Anchored to its own sentence rather than looked up loose. "45 of 64 sequences diverge"
    # appears further down the page, so a bare search for "64" is satisfied by a sentence that
    # has nothing to do with this bound -- the check would pass with the bound itself wrong.
    lo, hi = head_removal_range()
    sentence = f"removes {lo:.0f}% to {hi:.0f}% of the out-of-band tokens"
    ok = sentence in readme
    print(
        f"  {'fp32 head removal, on Qwen':<38} {sentence!r}  {'ok' if ok else 'NOT ON PAGE'}"
    )
    if not ok:
        failures.append(
            f"the README does not say {sentence!r}; measured {lo:.2f}% to {hi:.2f}%"
        )

    # A quantity spelled in words is invisible to every check above it.
    SPELLED = [
        "two thirds",
        "one third",
        "three quarters",
        "one quarter",
        "four fifths",
        "three fifths",
        "half of the out-of-band",
        "an order of magnitude of the",
    ]
    for phrase in SPELLED:
        if phrase in readme.lower():
            failures.append(
                f"the README says {phrase!r}; write the measured percentage instead, so this "
                "guard can check it"
            )

    if failures:
        print("\n  " + "\n  ".join(failures))
        return 1
    print("  every prose figure traces to a committed result file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
