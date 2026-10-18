from evals.report import render_report


def make_row(
    question,
    passed=True,
    answer="On 14 March 2024 [report.pdf, page 1].",
    kind="single_page",
    pages=(1,),
    found_pages=(1, 6),
):
    return {
        "type": kind,
        "input": question,
        "expected": "On 14 March 2024.",
        "answer": answer,
        "pages": list(pages),
        "found_pages": list(found_pages),
        "passed": passed,
        "reason": "" if passed else "The date is wrong.",
    }


ROWS = [
    make_row("When was the park commissioned?"),
    make_row("When did the park start operating?", kind="reworded"),
    make_row("Who is the mayor?", passed=False, kind="not_in_document"),
]


def render(rows=ROWS):
    return render_report(
        rows, model="gemini-2.5-flash", docs_only=True, min_pass_rate=0.9
    )


def test_report_shows_result_and_target():
    html = render()

    assert "2 of 3" in html
    assert "67%" in html
    assert "90%" in html


def test_report_lists_every_question_with_its_verdict():
    html = render()

    for row in ROWS:
        assert row["input"] in html
    assert "The date is wrong." in html


def test_report_counts_results_per_question_type():
    html = render()

    assert "Reworded question" in html
    assert "Not in the document" in html


def test_report_escapes_html_in_answers():
    """Answers come from an LLM and must never be rendered as markup."""
    html = render([make_row("Q?", answer="<script>alert(1)</script>")])

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_shows_whether_the_right_page_was_found():
    html = render(
        [
            make_row("Found?", pages=[1], found_pages=[1, 6]),
            make_row("Missed?", pages=[6], found_pages=[1, 2]),
            make_row("Control?", kind="not_in_document", pages=[], found_pages=[1]),
        ]
    )

    assert "✓ page 1" in html
    assert "✗ page 6 missing" in html
    assert html.count("<td class='small'>–</td>") == 1
