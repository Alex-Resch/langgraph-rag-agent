from datetime import UTC, datetime
from html import escape

TYPE_LABELS = {
    "single_page": "Answer on one page",
    "late_page": "Answer late in the document",
    "multi_page": "Answer combines several pages",
    "reworded": "Reworded question",
    "not_in_document": "Not in the document (control)",
}

STYLE = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2328;
       margin: 32px; font-size: 13px; line-height: 1.45; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 24px 0 8px; }
.meta { color: #59636e; margin-bottom: 16px; }
.summary { display: flex; gap: 12px; margin: 16px 0; }
.box { border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 14px; }
.box strong { display: block; font-size: 20px; }
.ok { color: #1a7f37; } .bad { color: #cf222e; }
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid #d1d9e0; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }
tr { break-inside: avoid; }
.small { color: #59636e; font-size: 11px; }
.why td { border-top: none; padding-top: 0; }
"""


def page_check(row: dict) -> str:
    """Show whether the search found the pages that hold the answer."""
    if not row["pages"]:
        return "–"
    missing = [page for page in row["pages"] if page not in row["found_pages"]]
    if missing:
        return "✗ " + ", ".join(f"page {page}" for page in missing) + " missing"
    return "✓ " + ", ".join(f"page {page}" for page in row["pages"])


def render_report(
    rows: list[dict], *, model: str, docs_only: bool, min_pass_rate: float
) -> str:
    """Render eval results as a printable HTML test report."""
    passed = sum(row["passed"] for row in rows)
    rate = passed / len(rows)
    met = rate >= min_pass_rate
    mode = "documents only, no web search" if docs_only else "documents with web search"

    type_rows = ""
    for kind, label in TYPE_LABELS.items():
        of_type = [row for row in rows if row["type"] == kind]
        if of_type:
            ok = sum(row["passed"] for row in of_type)
            type_rows += f"<tr><td>{label}</td><td>{ok} of {len(of_type)}</td></tr>"

    question_rows = ""
    for number, row in enumerate(rows, start=1):
        verdict = (
            '<span class="ok">✓ Passed</span>'
            if row["passed"]
            else '<span class="bad">✗ Failed</span>'
        )
        question_rows += (
            f"<tr><td>{number}</td>"
            f"<td>{escape(row['input'])}<div class='small'>{TYPE_LABELS.get(row['type'], row['type'])}</div></td>"
            f"<td>{escape(row['expected'])}</td><td>{escape(row['answer'])}</td>"
            f"<td class='small'>{page_check(row)}</td><td>{verdict}</td></tr>"
        )
        if not row["passed"]:
            question_rows += (
                f"<tr class='why'><td></td><td colspan='5' class='small'>"
                f"Why it failed: {escape(row['reason'])}</td></tr>"
            )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Chatbot test report</title>
<style>{STYLE}</style></head><body>
<h1>Chatbot test report</h1>
<div class="meta">{datetime.now(UTC).strftime("%d %B %Y")} · model {escape(model)} · mode: {mode}</div>
<div class="summary">
  <div class="box">Result<strong>{passed} of {len(rows)} correct ({rate:.0%})</strong></div>
  <div class="box">Target<strong>at least {min_pass_rate:.0%}</strong></div>
  <div class="box">Status<strong class="{"ok" if met else "bad"}">{"Target met" if met else "Target missed"}</strong></div>
</div>
<p>Every answer was compared with the expected answer by an independent AI judge (DeepEval).
Reworded questions check that different wording still finds the answer. Control questions
check that the assistant says so when the answer is not in the documents instead of guessing.</p>
<h2>Results by question type</h2>
<table><tr><th>Question type</th><th>Correct</th></tr>{type_rows}</table>
<h2>All questions</h2>
<table><colgroup><col style="width:4%"><col style="width:20%"><col style="width:17%">
<col style="width:37%"><col style="width:11%"><col style="width:11%"></colgroup>
<tr><th>#</th><th>Question</th><th>Expected answer</th><th>Assistant's answer</th>
<th>Right page found</th><th>Verdict</th></tr>{question_rows}</table>
</body></html>
"""
