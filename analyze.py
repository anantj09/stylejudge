"""Summarise all results/*.jsonl into tables (printed and saved to results/summary.md).

Metrics
  style_100     style bias b = P(judge prefers markdown side) - P(prefers prose side), as in the
                paper (Sec. 3.5); position-averaged over style (markdown in A) and style_ba
                (markdown in B). An unbiased judge scores 0 (it should say tie).
  mt_bench/llmbar
                agreement with human labels (exact match, ties included, as in the paper),
                Cohen's kappa, and paired comparison against B0 on the same items
                (bootstrap CI of the difference + exact McNemar test).
  mt_bench subsets
                "mismatch": exactly one response uses markdown. On this subset we also report
                the markdown-side preference of the judge AND of the human labels, so you can
                see whether the judge over-rewards markdown relative to people.
                "code": at least one response contains a fenced code block.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from normalize import has_markdown

ORDER = ["B0", "IGNORE", "S5", "S1", "STRIP", "KEEPCODE", "S5_STRIP", "PW", "PW_STRIP"]
rng = np.random.default_rng(0)


def load_data(name):
    return {json.loads(l)["id"]: json.loads(l) for l in open(f"data/{name}.jsonl")}


def boot_ci(values, n=2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def kappa(pred, gold, labels=("A", "B", "tie")):
    n = len(gold)
    if n == 0:
        return float("nan")
    po = sum(p == g for p, g in zip(pred, gold)) / n
    pe = sum((sum(p == l for p in pred) / n) * (sum(g == l for g in gold) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def md_side(row):
    """'A' or 'B' if exactly one response uses markdown, else None."""
    a, b = has_markdown(row["response_a"]), has_markdown(row["response_b"])
    return "A" if a and not b else "B" if b and not a else None


def md_pref(verdicts_and_sides):
    """P(pick markdown side) - P(pick plain side); ties count as neither."""
    v = [(1 if x == s else -1 if x in ("A", "B") else 0) for x, s in verdicts_and_sides]
    return (float(np.mean(v)) if v else float("nan")), v


def fmt_ci(m, ci):
    return f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def main():
    runs = defaultdict(dict)        # (model, dataset) -> condition -> {id: record}
    for p in glob.glob("results/*.jsonl"):
        recs = [json.loads(l) for l in open(p)]
        if recs:
            r0 = recs[0]
            runs[(r0["model"], r0["dataset"])][r0["condition"]] = {r["id"]: r for r in recs}

    lines = []
    out = lambda s="": (print(s), lines.append(s))

    for (model, dataset), conds in sorted(runs.items()):
        data = load_data(dataset)
        out(f"\n## {model} — {dataset}\n")
        names = [c for c in ORDER if c in conds]

        if dataset == "style_100":
            out("| condition | calls/pair | n | style bias (95% CI) | md in A | md in B | tie rate | invalid |")
            out("|---|---|---|---|---|---|---|---|")
            for c in names:
                recs = list(conds[c].values())
                pairs = [(r["verdict"], "A" if r["bias_type"] == "style" else "B") for r in recs]
                m, v = md_pref(pairs)
                inA = md_pref([p for p in pairs if p[1] == "A"])[0]
                inB = md_pref([p for p in pairs if p[1] == "B"])[0]
                ties = np.mean([r["verdict"] == "tie" for r in recs])
                inv = np.mean([r["verdict"] is None for r in recs])
                calls = np.mean([r.get("calls", 1) for r in recs])
                out(f"| {c} | {calls:.0f} | {len(recs)} | {fmt_ci(m, boot_ci(v))} | {inA:+.2f} | {inB:+.2f} | {ties:.0%} | {inv:.0%} |")
            out("\nUnbiased judge = 0. Positive = prefers markdown. 'md in A/B' split shows position effects.")

            # Paired view: each style_XXX pair and its style_ba mirror show the same two answers
            # in both orders. Which CONTENT does the judge pick each time?
            out("\n### Same pair, both orders (position-consistency)\n")
            out("| condition | pairs | md both times | prose both times | flips with position | involves a tie | "
                "position effect | style effect (swap-consistent) |")
            out("|---|---|---|---|---|---|---|---|")
            for c in names:
                recs = conds[c]
                both = [(i, data[i]["metadata"]["mirror_of"]) for i in recs
                        if data[i]["bias_type"] == "style_ba" and data[i]["metadata"].get("mirror_of") in recs]
                cnt = {"md": 0, "prose": 0, "flip": 0, "tie": 0}
                pos = []
                for ba_id, ab_id in both:
                    v1 = recs[ab_id]["verdict"]     # markdown shown in A
                    v2 = recs[ba_id]["verdict"]     # markdown shown in B
                    if "tie" in (v1, v2) or None in (v1, v2):
                        cnt["tie"] += 1
                    elif v1 == "A" and v2 == "B":
                        cnt["md"] += 1
                    elif v1 == "B" and v2 == "A":
                        cnt["prose"] += 1
                    else:
                        cnt["flip"] += 1           # same slot both times = position decided it
                    pos += [{"A": 1, "B": -1}.get(v1, 0), {"A": 1, "B": -1}.get(v2, 0)]
                n = len(both)
                if n:
                    out(f"| {c} | {n} | {cnt['md']} | {cnt['prose']} | {cnt['flip']} | {cnt['tie']} | "
                        f"{np.mean(pos):+.2f} | {(cnt['md'] - cnt['prose']) / n:+.2f} |")
            out("\nposition effect = P(pick slot A) − P(pick slot B) over both orders (0 = none). "
                "style effect counts only pairs where the judge picked the same content in both orders, "
                "i.e. the paper's position-swap rule (S1) applied to data we already have.")
            continue

        # ---- human-labelled benchmarks ----
        base = conds.get("B0")
        out("| condition | calls/pair | n | agreement (95% CI) | kappa | Δ vs B0 (95% CI) | McNemar p | tie rate | invalid |")
        out("|---|---|---|---|---|---|---|---|---|")
        for c in names:
            recs = conds[c]
            ids = sorted(recs)
            correct = [recs[i]["verdict"] == recs[i]["human"] for i in ids]
            acc = float(np.mean(correct))
            ci = boot_ci(correct)
            k = kappa([recs[i]["verdict"] for i in ids], [recs[i]["human"] for i in ids])
            delta, pval = "—", "—"
            if base is not None and c != "B0":
                common = sorted(set(ids) & set(base))
                d = [int(recs[i]["verdict"] == recs[i]["human"]) - int(base[i]["verdict"] == base[i]["human"])
                     for i in common]
                b = sum(x == 1 for x in d)
                cc = sum(x == -1 for x in d)
                dci = boot_ci(d)
                delta = f"{np.mean(d) * 100:+.1f} pp [{dci[0] * 100:+.1f}, {dci[1] * 100:+.1f}]"
                pval = f"{mcnemar_exact(b, cc):.3f}"
            ties = np.mean([recs[i]["verdict"] == "tie" for i in ids])
            inv = np.mean([recs[i]["verdict"] is None for i in ids])
            calls = np.mean([recs[i].get("calls", 1) for i in ids])
            out(f"| {c} | {calls:.0f} | {len(ids)} | {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] | {k:.3f} | {delta} | {pval} | {ties:.0%} | {inv:.0%} |")

        if dataset == "mt_bench_400":
            mism = {i for i, r in data.items() if md_side(r)}
            code = {i for i, r in data.items() if "```" in r["response_a"] + r["response_b"]}
            out(f"\nSubsets: mismatch = exactly one side uses markdown (n={len(mism)}); "
                f"code = a fenced code block appears (n={len(code)}).\n")
            out("| condition | agree: mismatch | agree: rest | agree: code | md-side pref: judge (95% CI) | humans | "
                "excess over humans (95% CI) | Δ excess vs B0 (95% CI) |")
            out("|---|---|---|---|---|---|---|---|")
            mism_ids = sorted(mism)
            base_v = None
            if base is not None:
                _, base_v = md_pref([(base[i]["verdict"], md_side(data[i])) for i in mism_ids if i in base])
            for c in names:
                recs = conds[c]
                sub = lambda S: np.mean([recs[i]["verdict"] == recs[i]["human"] for i in recs if i in S]) if any(i in S for i in recs) else float("nan")
                rest = set(recs) - mism
                ids_m = [i for i in mism_ids if i in recs]
                jm, jv = md_pref([(recs[i]["verdict"], md_side(data[i])) for i in ids_m])
                hm, hv = md_pref([(recs[i]["human"], md_side(data[i])) for i in ids_m])
                excess = [j - h for j, h in zip(jv, hv)]          # paired per item
                dcol = "—"
                if base_v is not None and c != "B0" and len(base_v) == len(jv):
                    d = [j - b for j, b in zip(jv, base_v)]
                    dci = boot_ci(d)
                    dcol = f"{np.mean(d):+.2f} [{dci[0]:+.2f}, {dci[1]:+.2f}]"
                out(f"| {c} | {sub(mism):.1%} | {sub(rest):.1%} | {sub(code):.1%} | {fmt_ci(jm, boot_ci(jv))} | "
                    f"{hm:+.2f} | {fmt_ci(np.mean(excess), boot_ci(excess))} | {dcol} |")
            out("\nmd-side pref = P(pick markdown side) − P(pick plain side) on mismatch pairs. "
                "If the judge's value is far above the humans', it over-rewards markdown. "
                "Δ excess vs B0 < 0 with a CI excluding 0 means the method significantly reduces that over-reward.")

    os.makedirs("results", exist_ok=True)
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved results/summary.md")


if __name__ == "__main__":
    main()
