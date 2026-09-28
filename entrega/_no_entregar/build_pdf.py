import re
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 1.6cm 1.7cm; }
body { font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #111; text-align: justify; }
h1 { color: #1f3864; font-size: 16pt; margin: 0 0 6px; }
h2 { color: #1f3864; font-size: 13pt; border-bottom: 1.5px solid #1f3864; padding-bottom: 2px; margin: 14px 0 6px; }
h3 { font-size: 11pt; margin: 10px 0 4px; }
p { margin: 4px 0; } ul, ol { margin: 4px 0; padding-left: 20px; } li { margin: 2px 0; }
code { font-family: Consolas, monospace; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 9pt; text-align: left; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3px 6px; } th { background: #dde3ee; text-align: center; }
img { display: block; margin: 4px auto 0; } img.wide { width: 100%; } img.cm { width: 52%; }
"""


def main():
    md = (HERE / "informe_tecnico.md").read_text(encoding="utf-8")
    md = re.sub(r"^---\s*$", "", md, flags=re.M)
    md = re.sub(r"^(\*\*(Materia|Integrantes):\*\*.*)$", r"\1  ", md, flags=re.M)
    md = md.replace("](reports/", "](../reports/")
    body = markdown.markdown(md, extensions=["tables"])
    body = (
        body.replace('<img alt="Curvas', '<img class="wide" alt="Curvas')
        .replace('<img alt="Matriz', '<img class="cm" alt="Matriz')
        .replace('<img alt="Predicción', '<img class="wide" alt="Predicción')
    )
    html = HERE / "informe_tecnico.html"
    html.write_text(
        f"<!doctype html><html lang='es'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    pdf = ROOT / "informe_tecnico.pdf"
    subprocess.run(
        [EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
        check=True,
    )
    print(f"PDF generado: {pdf}")


if __name__ == "__main__":
    sys.exit(main())
