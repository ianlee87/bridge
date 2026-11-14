"""Score Bridge on the cases in test_phrases.txt, from text only (no audio).

For each case it does what main.py does after speech-to-text:
  urgent keyword check first, then Bonsai (with the first-person filter).

Scoring is a simple keyword overlap: what fraction of the expected sentence's
content words (ignoring words like "I", "the", "please") appear in an option.
The best of the 3 options counts:
    HIT    >= 0.5
    UNSURE  0.25 - 0.5   (a person should judge these)
    MISS   < 0.25

The 14 normal cases run twice: with the pointed-at object, and without it.

Run:  .venv/bin/python eval.py
"""
import re
import time

from rich.console import Console
from rich.table import Table

from brain import Brain, first_person_filter
from fragments import check_urgent

console = Console()

HIT_AT = 0.5
UNSURE_AT = 0.25

# Small words that don't carry meaning, ignored when comparing sentences.
STOP_WORDS = set("""
a an the i i'd i'm im me my to of in on for and or is it it's its be am are
was can could would will you your please some that this just with by at so
do does like want yes no not
""".split())


def load_cases(path="test_phrases.txt"):
    cases = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        caregiver, frags, pointing, expected = [p.strip() for p in line.split("|")]
        cases.append({
            "caregiver": None if caregiver == "none" else caregiver.strip('"'),
            "fragments": [f.strip() for f in frags.split("...") if f.strip()],
            "pointing": None if pointing == "none" else pointing,
            "expected": expected,
        })
    return cases


def content_words(sentence):
    words = re.findall(r"[a-z']+", sentence.lower().replace("’", "'"))
    # Very simple stemming: drop trailing "s" so "blankets" matches "blanket".
    # (It mangles some words, but both sentences get the same treatment.)
    return {w.rstrip("s") for w in words if w not in STOP_WORDS}


def overlap(option, expected):
    want = content_words(expected)
    if not want:
        return 0.0
    return len(want & content_words(option)) / len(want)


def verdict(score):
    if score >= HIT_AT:
        return "[green]HIT[/]"
    if score >= UNSURE_AT:
        return "[yellow]UNSURE[/]"
    return "[red]MISS[/]"


def run_case(brain, case, use_pointing):
    """Returns a dict with options, best score, and notes for one case."""
    reason = check_urgent(case["fragments"])
    if reason:
        # Wrong alert on a normal case: Bonsai is skipped, so this is a miss.
        return {"options": [], "score": 0.0, "note": f"false URGENT ({reason})", "ms": 0}
    t0 = time.perf_counter()
    result = brain.suggest(case["fragments"], case["caregiver"],
                           case["pointing"] if use_pointing else None)
    elapsed = (time.perf_counter() - t0) * 1000
    options = result["options"]
    score = max((overlap(o, case["expected"]) for o in options), default=0.0)
    note = []
    if result["dropped"]:
        note.append(f"filter dropped {len(result['dropped'])}")
    if result["retried"]:
        note.append("asked twice")
    return {"options": options, "dropped": result["dropped"], "score": score,
            "note": ", ".join(note), "ms": elapsed}


def main():
    brain = Brain()
    t0 = time.perf_counter()
    brain.warm_up()
    console.print(f"[green]Bonsai warm[/] [dim]({(time.perf_counter() - t0) * 1000:.0f} ms)[/]\n")

    cases = load_cases()
    normal = [c for c in cases if c["expected"] != "URGENT"]
    urgent = [c for c in cases if c["expected"] == "URGENT"]

    # ---- Urgent cases: the keyword alert must fire, Bonsai must not be called ----
    for c in urgent:
        reason = check_urgent(c["fragments"])
        status = "[green]FIRED[/]" if reason else "[red]DID NOT FIRE[/]"
        console.print(f"URGENT case {' ... '.join(c['fragments'])!r}: {status}"
                      + (f" (trigger: {reason})" if reason else ""))
    console.print()

    # ---- 14 normal cases, with and without pointing_at ----
    results = {True: [], False: []}
    for use_pointing in (True, False):
        label = "WITH pointing_at" if use_pointing else "WITHOUT pointing_at"
        console.rule(f"[bold]{label}")
        for i, c in enumerate(normal, start=1):
            r = run_case(brain, c, use_pointing)
            results[use_pointing].append(r)
            # The filter would also drop the EXPECTED sentence? Worth knowing.
            expected_ok = bool(first_person_filter([c["expected"]], c["fragments"]))
            console.print(
                f"[bold]{i:>2}.[/] {' ... '.join(c['fragments'])}"
                f"  [dim]| asked: {c['caregiver'] or '-'}"
                f" | pointing: {(c['pointing'] if use_pointing else None) or '-'}[/]")
            console.print(f"    expected: {c['expected']}"
                          + ("" if expected_ok else "  [magenta](the filter would drop this)[/]"))
            for o in r["options"]:
                console.print(f"    - {o}  [dim]({overlap(o, c['expected']):.2f})[/]")
            for d in r.get("dropped", []):
                console.print(f"    [dim]x filtered: {d}[/]")
            console.print(f"    best {r['score']:.2f} {verdict(r['score'])}"
                          f"  [dim]{r['ms']:.0f} ms {r['note']}[/]\n")

    # ---- Side-by-side summary ----
    table = Table(title="Summary (keyword overlap, best of options)")
    table.add_column("#", justify="right")
    table.add_column("Fragments")
    table.add_column("With pointing", justify="center")
    table.add_column("Without", justify="center")
    for i, c in enumerate(normal):
        w, wo = results[True][i], results[False][i]
        table.add_row(str(i + 1), " ... ".join(c["fragments"]),
                      f"{w['score']:.2f} {verdict(w['score'])}",
                      f"{wo['score']:.2f} {verdict(wo['score'])}")

    def rate(rs, at):
        return sum(r["score"] >= at for r in rs)

    n = len(normal)
    for use_pointing in (True, False):
        rs = results[use_pointing]
        avg_ms = sum(r["ms"] for r in rs) / len(rs)
        results[use_pointing] = (rate(rs, HIT_AT), rate(rs, UNSURE_AT) - rate(rs, HIT_AT), avg_ms)
    (h1, u1, ms1), (h2, u2, ms2) = results[True], results[False]
    table.add_section()
    table.add_row("", "[bold]HIT rate[/]", f"[bold]{h1}/{n} ({h1 / n:.0%})[/]",
                  f"[bold]{h2}/{n} ({h2 / n:.0%})[/]")
    table.add_row("", "UNSURE (judge by hand)", str(u1), str(u2))
    table.add_row("", "avg Bonsai time", f"{ms1:.0f} ms", f"{ms2:.0f} ms")
    console.print(table)


if __name__ == "__main__":
    main()
