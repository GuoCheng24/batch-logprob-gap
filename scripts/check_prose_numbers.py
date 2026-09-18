"""The numbers the README states in prose must come from a committed result file.

The tables are safe: `build_tables.py` regenerates them and CI diffs the result. The prose is not.
One sentence -- "enabling vLLM's batch-invariant kernels does not shrink it: 8.0% to 7.1%, while
changing the trainer's precision goes to 3.7% in one step" -- stood for days with no file behind it
in this repository; the measurement existed, but only on the machine that ran it. An audit found it,
not a test. This guard re-derives each such figure from `results/` so the next one cannot survive.

    python scripts/check_prose_numbers.py
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def pct(arm, data):
    return 100 * data["arms"][arm]["out"] / data["n_tokens"]


def hf_rate(model, key="bf16|b1_vs_b8"):
    r = json.load(open(RESULTS / f"hfonly_{model}.json"))["results"][key]
    return 100 * r["out"] / r["n"]


def flips(model):
    d = json.load(open(RESULTS / f"grpo_{model}.json"))
    return 100 * d["flips"] / d["n"]


def main() -> int:
    kern = json.load(open(RESULTS / "kernel_arms.json"))
    dtype = json.load(open(RESULTS / "dtype_pythia-410m.json"))["results"]
    pack = json.load(open(RESULTS / "pack_pythia-410m.json"))["results"]

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
        ),  # refresh_readme.py 覆盖
        ("clip-status flips, smallest", flips("Qwen2.5-0.5B-Instruct"), "1.7"),
        ("clip-status flips, largest", flips("pythia-410m"), "10.1"),
    ]
    readme = (ROOT / "README.md").read_text()
    failures = []
    for label, got, written in checks:
        if written is None:
            print(f"  {label:<38} {got:>6.2f}%  (refresh_readme.py 覆盖)")
            continue
        places = len(written.split(".")[1])
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
    if failures:
        print("\n  " + "\n  ".join(failures))
        return 1
    print("  every prose figure traces to a committed result file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
