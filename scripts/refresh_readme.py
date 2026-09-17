"""Rewrite the README statements that depend on which models are in results/.

Every number here is read from results/*.json at run time. Anchors are asserted so a
changed README fails loudly instead of being silently left stale."""
import json, glob, os, re, pathlib
R = pathlib.Path(__file__).resolve().parent.parent
rows = {}
for f in glob.glob(str(R / "results" / "hfonly_*.json")):
    d = json.load(open(f)); r = d["results"]["bf16|b1_vs_b8"]
    rows[d["model"].split("/")[-1]] = (100 * r["out"] / r["n"], r["n"])
assert rows, "no hfonly results"
lo = min(v[0] for v in rows.values()); hi = max(v[0] for v in rows.values())
nmin = min(v[1] for v in rows.values()); nmax = max(v[1] for v in rows.values())
arch = {"pythia": "GPT-NeoX", "Qwen2.5": "Qwen2", "granite": "Granite MoE", "Agents": "Agents-A1"}
fams = {next(a for k, a in arch.items() if m.startswith(k)) for m in rows}
def params_b(m):
    """Approximate parameter count in billions from the model name."""
    mm = re.search(r"(\d+\.?\d*)\s*[mM]\b", m)
    if mm and "pythia" in m: return float(mm.group(1)) / 1000
    b = re.search(r"(\d+\.?\d*)\s*[bB]\b", m)
    return float(b.group(1)) if b else 0.0
sizes = sorted(rows, key=lambda m: rows[m][0])
lowest = sizes[0]; largest = max(rows, key=params_b)
assert params_b("pythia-410m") == 0.41 and params_b("Qwen2.5-7B-Instruct") == 7.0 and params_b("Agents-A1-4B") == 4.0

p = R / "README.md"; s = p.read_text()
def sub(pattern, new):
    """Regex anchor so the script is idempotent: it matches both the original wording and
    its own previous output, and refuses to run if the anchor is not exactly once."""
    global s
    s, k = re.subn(pattern, new, s, flags=re.S)
    assert k == 1, f"anchor matched {k} times: {pattern[:60]!r}"

sub(r"(On four architectures|Across \d+ architectures and \d+ models) the resulting",
    f"Across {len(fams)} architectures and {len(rows)} models the resulting")
sub(r"for \*\*[\d.]+% to [\d.]+%\*\* of tokens", f"for **{lo:.1f}% to {hi:.1f}%** of tokens")
sub(r"[\d,]+(?: to [\d,]+)? tokens per\ncell\.", f"{nmin:,} to {nmax:,} tokens per\ncell.")
m84 = re.search(r"- \*\*Small models\.\*\*.*?(?=\n- \*\*|\n\n)", s, re.S)
assert m84, "Small-models paragraph not found"
old84 = m84.group(0)
new84 = (f"- **Small models.** The largest here is {largest.replace('-Instruct','')}; the production report "
         f"was a 30B MoE. Within the Qwen2.5 family the rate drifts from "
         + ", ".join(f"{rows[m][0]:.1f}%" for m in ["Qwen2.5-0.5B-Instruct","Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct","Qwen2.5-7B-Instruct"] if m in rows)
         + f" ({', '.join(m.replace('Qwen2.5-','').replace('-Instruct','') for m in ['Qwen2.5-0.5B-Instruct','Qwen2.5-1.5B-Instruct','Qwen2.5-3B-Instruct','Qwen2.5-7B-Instruct'] if m in rows)}), "
         f"so size attenuates it slowly within a family, while the spread across families is far larger than "
         f"the spread across sizes. The lowest rate in the set is {lowest.replace('-Instruct','')}. "
         f"What a 30B MoE does is not something this harness can say.")
s = s.replace(old84, new84)
# splice the embedded tables from TABLES.md so README never lags the data
tb = (R / "TABLES.md").read_text()
def section(tag):
    m = re.search(rf"## {tag} .*?\n\n(\|.*?)(?=\n\n|\Z)", tb, re.S)
    assert m, f"section {tag} missing in TABLES.md"
    return m.group(1).strip()
for tag in ("T1", "T2", "T7", "T10"):
    a, b = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    assert s.count(a) == 1 and s.count(b) == 1, f"README lacks {tag} markers"
    s = s[:s.index(a) + len(a)] + "\n" + section(tag) + "\n" + s[s.index(b):]
p.write_text(s)
print(f"range {lo:.1f}-{hi:.1f}%  families {len(fams)}  models {len(rows)}  n {nmin}-{nmax}  largest {largest}  lowest {lowest}")
