"""Assemble paper/report.md from paper/template.md and TABLES.md: every `@Tn` line is replaced by that table's
pipe table verbatim, so the report can never quote a table the JSON did not produce. Then check that every
number in the prose also occurs in TABLES.md, README.md or paper/numbers.json.

    python paper/build.py
    pandoc paper/report.md -o paper/report.pdf --pdf-engine=tectonic   # the committed PDF, six pages

The second line is the whole of how paper/report.pdf is produced; it was not written down anywhere
until a change to the prose left the PDF a version behind the page it was built from."""

import re, json, pathlib, sys

R = pathlib.Path(__file__).resolve().parent.parent
tables = (R / "TABLES.md").read_text()
tpl = (R / "paper/template.md").read_text()


def section(tag):
    m = re.search(
        rf"^## {re.escape(tag)} .*?\n\n(\|.*?)(?=\n\n|\Z)", tables, re.S | re.M
    )
    assert m, tag
    return m.group(1).strip()


out = re.sub(r"^@(T\d+b?)$", lambda m: section(m.group(1)), tpl, flags=re.M)
(R / "paper/report.md").write_text(out)
# number audit: prose tokens (outside tables) must exist somewhere in the sources
prose = "\n".join(l for l in out.split("\n") if not l.startswith("|"))
prose = re.sub(
    r"arXiv:\d{4}\.\d{5}|#\d+|\d+B(?:-A\d+B)?|SM89|4090|20\d\d|\d+ ?×|\\times10\^\{-6\}",
    " ",
    prose,
)
src = tables + (R / "README.md").read_text() + (R / "paper/numbers.json").read_text()
# the end-to-end run of the TRL change is a separate result file; quote it from the file, not from memory
import glob

for f in glob.glob(str(R / "results/e2e_*.json")):
    d = json.load(open(f))
    src += " " + " ".join(
        f"{sum(d['dlogp_mean'])/len(d['dlogp_mean']):.3f}" for _ in [0]
    )
WHITELIST = {"0.1", "2.3"}  # the clip band half-width epsilon, the page margin
srcn = set(re.findall(r"\d[\d,]*\.?\d*", src))
bad = []
for tok in re.findall(r"\d[\d,]*\.?\d*", prose):
    t = tok.rstrip(".")
    if (
        t
        in {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "16",
            "0",
            "150",
            "30",
            "35",
            "64",
            "45",
        }
        or t in WHITELIST
    ):
        continue
    if t not in srcn and t.replace(",", "") not in srcn:
        bad.append(t)
print(
    "report.md assembled;",
    "prose numbers not found in sources:",
    sorted(set(bad)) or "none",
)
sys.exit(1 if bad else 0)
