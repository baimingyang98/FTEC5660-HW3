"""Offline checks for hw3.py: no API key and no model calls (a keyword-based fake model stands in).

Run from the repository root:  python tests/test_offline.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hw3  # noqa: E402

PUBLIC = ROOT / "public_test"
TRUTH = json.loads((PUBLIC / "ground_truth.json").read_text(encoding="utf-8"))
RESULTS = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS["passed" if condition else "failed"] += 1
    print(f"{'ok  ' if condition else 'FAIL'} {name}" + (f"  ({detail})" if detail and not condition else ""))


class FakeModel:
    """Answers the agent's prompts with simple keyword rules, so the plumbing can be tested offline."""

    NEGATIVE = ("quit", "overtime", "strike", "morale is low", "turnover", "underpaid")
    BAD_NEWS = ("fell", "fined", "lost", "cut its", "warned", "recall", "lawsuit")

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def _count(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        if "Fisher's point 7" in prompt:
            self._count("labour")
            stances = {}
            for line in prompt.splitlines():
                match = re.match(r"^(current_employee|former_employee|customer|competitor|supplier): (.*)$", line)
                if not match:
                    continue
                answer = match.group(2).lower()
                if "comment" in answer:
                    stances[match.group(1)] = "no_information"
                elif any(word in answer for word in self.NEGATIVE):
                    stances[match.group(1)] = "negative"
                else:
                    stances[match.group(1)] = "positive"
            verdict = "FAIL" if "negative" in stances.values() else "PASS"
            return SimpleNamespace(content="```json\n" + json.dumps({"stances": stances, "verdict": verdict}) + "\n```")
        if "Fisher's point 15" in prompt:
            self._count("news")
            bad = [int(m.group(1)) for m in re.finditer(r"^(\d+)\. \d{4}-\d{2}-\d{2}: (.*)$", prompt, re.M)
                   if any(word in m.group(2).lower() for word in self.BAD_NEWS)]
            return SimpleNamespace(content=json.dumps({"bad_news": bad, "reason": "keywords"}))
        if "Write a short summary" in prompt:
            self._count("summary")
            return SimpleNamespace(content="A short summary.")
        self._count("other")
        return SimpleNamespace(content='{"ok": true}')


class BadRequestError(Exception):
    """Same class name as the OpenAI client's 400 error, which FisherAgent.ask() looks for."""


class RejectsThinking(FakeModel):
    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        raise BadRequestError("unknown parameter: thinking")


class Broken:
    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        raise RuntimeError("API down")


def public_companies() -> dict[str, list]:
    return {path.name: hw3.make_tools(path) for path in hw3.company_folders(PUBLIC)}


def score(analyses: dict) -> tuple[int, int, float]:
    chosen = hw3.check_chosen_points(hw3.CHOSEN_POINTS)
    correct = sum(
        hw3.normalize_points(analyses[name]["points"], chosen)[p] == TRUTH[name]["points"][str(p)]
        for name in TRUTH for p in chosen
    )
    return correct, len(TRUTH) * len(chosen), hw3.balanced_point_accuracy(analyses, TRUTH, list(TRUTH), chosen)


def test_helpers() -> None:
    check("parse_json reads a fenced reply", hw3.parse_json('```json\n{"a": 1}\n```') == {"a": 1})
    try:
        hw3.parse_json("no json here")
        check("parse_json rejects text without JSON", False)
    except ValueError:
        check("parse_json rejects text without JSON", True)
    check("iso_day keeps a calendar date", hw3.iso_day("2025-10-31") == "2025-10-31")
    check("iso_day rejects a non-date", hw3.iso_day("31 Oct 2025") is None and hw3.iso_day("2025-02-30") is None)
    news = "# X: news\n\n- 2025-10-31: Regulators fined X USD 7 million.\n- 2026-02-05: X signed a deal.\nnot an item"
    items = hw3.parse_news(news)
    check("parse_news finds the dated items", [i["date"] for i in items] == ["2025-10-31", "2026-02-05"])
    check("parse_news keeps the headline", items[0]["text"] == "Regulators fined X USD 7 million.")
    check("chosen points are 5, 7 and 15", sorted(hw3.CHOSEN_POINTS) == [5, 7, 15])


def test_sql() -> None:
    tools = {t.name: t for t in hw3.make_tools(PUBLIC / "hollins_group")}
    row = hw3.sql_rows(tools["sql_query"], hw3.MARGIN_SQL)[0]
    check("margin SQL uses the latest full fiscal year", row["fiscal_year"] == "2025", str(row))
    check("margin SQL compares with the own industry median",
          row["industry"] == "medical devices" and row["industry_median_pct"] == "16.80", str(row))
    check("margin SQL gives a verdict", row["verdict"] in ("PASS", "FAIL"), str(row))
    tools = {t.name: t for t in hw3.make_tools(PUBLIC / "copperfield_technologies")}
    sales = hw3.sql_rows(tools["sql_query"], hw3.SELLS_SQL.format(day="2026-03-09", window=hw3.SELL_WINDOW_DAYS))
    check("sales window SQL counts days with julianday",
          [s["days_before"] for s in sales] == ["28"], str(sales))
    sales = hw3.sql_rows(tools["sql_query"], hw3.SELLS_SQL.format(day="2025-08-18", window=hw3.SELL_WINDOW_DAYS))
    check("sales window SQL ignores buys", sales == [], str(sales))
    try:
        hw3.sql_rows(tools["sql_query"], "SELECT nope FROM nowhere")
        check("sql_rows raises on an SQL error", False)
    except RuntimeError:
        check("sql_rows raises on an SQL error", True)


def test_public_with_fake_model() -> None:
    fake = FakeModel()
    companies = public_companies()
    analyses = asyncio.run(hw3.analyze_companies(hw3.FisherAgent(model=fake), companies))
    correct, total, balanced = score(analyses)
    check("every public company has an entry", set(analyses) == set(TRUTH))
    check("rules match all public answers on points 5, 7, 15", correct == total, f"{correct}/{total}")
    check("balanced score is 100%", balanced == 1.0, f"{balanced:.1%}")
    used = set(hw3.INTERVIEWS_USED.values())
    check("each company used exactly 2 interviews", used == {2}, str(hw3.INTERVIEWS_USED))
    check("one labour call per company", fake.calls.get("labour") == len(TRUTH), str(fake.calls))
    check("one summary per company", fake.calls.get("summary") == len(TRUTH), str(fake.calls))
    check("news is sent to the model only when an insider sold before it",
          0 < fake.calls.get("news", 0) < len(TRUTH), str(fake.calls))
    for name, truth in TRUTH.items():
        p15 = analyses[name]["points"][15]
        if truth["points"]["15"] == "FAIL":
            check(f"{name}: point 15 FAIL", p15 == "FAIL")


def test_failures_are_contained() -> None:
    analyses = asyncio.run(hw3.analyze_companies(hw3.FisherAgent(model=Broken()), public_companies()))
    check("a broken model still gives an entry per company", set(analyses) == set(TRUTH))
    complete = all(set(e["points"]) == {5, 7, 15} and all(v in ("PASS", "FAIL") for v in e["points"].values())
                   for e in analyses.values())
    check("a broken model still gives a verdict on every point", complete)
    point5 = sum(analyses[n]["points"][5] == TRUTH[n]["points"]["5"] for n in TRUTH)
    check("point 5 needs no model", point5 == len(TRUTH), f"{point5}/{len(TRUTH)}")
    check("summaries fall back to the facts", all(e["summary"] for e in analyses.values()))

    agent = hw3.FisherAgent(model=None)
    agent.model, agent.model_error = None, "no key"
    analyses = asyncio.run(hw3.analyze_companies(agent, public_companies()))
    point5 = sum(analyses[n]["points"][5] == TRUTH[n]["points"]["5"] for n in TRUTH)
    check("without any model, point 5 is still computed", point5 == len(TRUTH), f"{point5}/{len(TRUTH)}")


def test_thinking_switch() -> None:
    original = hw3.make_model
    hw3.make_model = lambda thinking_off=True: FakeModel()
    try:
        agent = hw3.FisherAgent(model=RejectsThinking())
        reply = asyncio.run(agent.ask("hello"))
        check("a rejected thinking switch is retried without it", reply == '{"ok": true}' and not agent.thinking_off)
    finally:
        hw3.make_model = original


def test_no_hard_coding() -> None:
    source = (ROOT / "hw3.py").read_text(encoding="utf-8")
    helpers = source.split("# My agent")[1].split("# Choose any 3 of the 15 points")[0]
    bodies = "".join(part.split("\n\n\n")[0] for part in source.split("### YOUR CODE HERE")[1:])
    mine = helpers + bodies
    names = [p.name for p in hw3.company_folders(PUBLIC)] + [v["company"] for v in TRUTH.values()]
    leaked = [n for n in names if n.lower() in mine.lower()]
    check("no public company name in my code", not leaked, str(leaked))
    check("my code does not read the answer key", "ground_truth" not in mine)
    check("my code reads data only through the tools",
          all(word not in mine for word in ("company.db", "interviews.dat", "sqlite3", "zlib")))


if __name__ == "__main__":
    for test in (test_helpers, test_sql, test_public_with_fake_model, test_failures_are_contained,
                 test_thinking_switch, test_no_hard_coding):
        print(f"\n== {test.__name__}")
        test()
    print(f"\n{RESULTS['passed']} passed, {RESULTS['failed']} failed")
    sys.exit(1 if RESULTS["failed"] else 0)
