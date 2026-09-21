"""paper/report.pdf must be built from the paper/report.md that is committed next to it.

The PDF is a binary artefact. Nothing stopped it from going a version behind the prose it was
built from -- and it did: a bound in section 4 was corrected on the page while the committed PDF
still carried the old wording, and every check in this repository passed.

Every percentage and every signed decimal the report's prose states must appear in the PDF's
extracted text. Table cells are excluded: they come from TABLES.md through build.py, which CI
already diffs, and pdftotext reflows table rows in ways that are not worth matching.

    python paper/check_pdf_current.py        # needs pdftotext (poppler-utils)
"""

import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "paper/report.md"
PDF = ROOT / "paper/report.pdf"

# a percentage, or a signed decimal such as $-0.013$ -- the two shapes the prose uses for a
# result. The sign has to touch the digit and not follow another dash, or the second hyphen of
# an em-dash written "--" turns "-- 0.07% and below" into a signed number that is not there.
TOKEN = re.compile(r"\d+\.\d+%|\d+%|(?<![-\w])[-+−]\d*\.\d+")


def main() -> int:
    if not shutil.which("pdftotext"):
        print("  pdftotext not found; install poppler-utils")
        return 1
    prose = "\n".join(
        ln for ln in MD.read_text().splitlines() if not ln.startswith("|")
    )
    text = subprocess.run(
        ["pdftotext", str(PDF), "-"], capture_output=True, text=True, check=True
    ).stdout
    # the PDF renders a TeX minus, and pdftotext gives it back as U+2212
    text_n = text.replace("\u2212", "-")

    def present(tok):
        """Anchored, not substring: "64%" must not be satisfied by "8.64%" elsewhere in the PDF."""
        t = tok.replace("\u2212", "-")
        return re.search(r"(?<![\d.])" + re.escape(t) + r"(?!\d)", text_n) is not None

    wanted = sorted({m.group(0) for m in TOKEN.finditer(prose)})
    missing = [t for t in wanted if not present(t)]
    print(f"  {len(wanted)} figures stated in the report's prose")
    for t in missing:
        print(f"    missing from the PDF: {t}")
    if missing:
        print(
            "\n  paper/report.pdf is behind paper/report.md. Rebuild it:\n"
            "    pandoc paper/report.md -o paper/report.pdf --pdf-engine=tectonic"
        )
        return 1
    print("  the committed PDF carries every figure the report's prose states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
