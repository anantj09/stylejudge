"""Run one judging condition on one dataset. Resumable: finished items are skipped.

Conditions (judge calls per pair in brackets):
  B0          naive prompt, raw responses                       [1] paper baseline
  IGNORE      naive prompt + "ignore formatting" instruction    [1] paper's 2nd recommendation
  S5          chain-of-thought prompt, raw responses            [1] paper's best style reducer
  S1          naive prompt, both orders; disagree -> tie         [2] paper's position swap
  STRIP       naive prompt, markdown stripped from both sides   [1] ours (normalisation)
  KEEPCODE    naive prompt, markdown stripped except code       [1] ours (variant)
  S5_STRIP    chain-of-thought + stripped                       [1] ours (combination)
  PW          pointwise: score each response alone, compare     [2] ours (no A/B slots)
  PW_STRIP    pointwise on markdown-stripped responses          [2] ours (pointwise + normalisation)

Examples:
  python run.py --condition B0 --dataset style_100 --model qwen2.5:7b-instruct
  python run.py --condition STRIP --dataset mt_bench_400 --model qwen2.5:7b-instruct --limit 50
  python run.py --condition B0 --dataset style_100 --model mock          # offline test
"""
import argparse
import json
import os
import time

from judge import POINTWISE, PROMPTS, MockJudge, OllamaJudge, parse_score, parse_verdict
from normalize import NORMALISERS

# condition: (mode, prompt, normaliser)
CONDITIONS = {
    "B0": ("pair", "naive", "none"),
    "IGNORE": ("pair", "ignore_format", "none"),
    "S5": ("pair", "cot", "none"),
    "S1": ("swap", "naive", "none"),
    "STRIP": ("pair", "naive", "strip"),
    "KEEPCODE": ("pair", "naive", "keep_code"),
    "S5_STRIP": ("pair", "cot", "strip"),
    "PW": ("point", None, "none"),
    "PW_STRIP": ("point", None, "strip"),
}
FLIP = {"A": "B", "B": "A", "tie": "tie", None: None}


def judge_item(judge, mode, template, q, a, b):
    """Return (verdict, list of raw outputs)."""
    if mode == "pair":
        out = judge(template.format(question=q, response_a=a, response_b=b))
        return parse_verdict(out["text"]), [out]
    if mode == "swap":                                   # paper's S1 rule
        o1 = judge(template.format(question=q, response_a=a, response_b=b))
        o2 = judge(template.format(question=q, response_a=b, response_b=a))
        v1, v2 = parse_verdict(o1["text"]), FLIP[parse_verdict(o2["text"])]
        if v1 is None or v2 is None:
            return None, [o1, o2]
        return (v1 if v1 == v2 else "tie"), [o1, o2]
    # pointwise
    oa = judge(POINTWISE.format(question=q, response=a))
    ob = judge(POINTWISE.format(question=q, response=b))
    sa, sb = parse_score(oa["text"]), parse_score(ob["text"])
    if sa is None or sb is None:
        return None, [oa, ob]
    return ("A" if sa > sb else "B" if sb > sa else "tie"), [oa, ob]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--dataset", required=True, choices=["style_100", "mt_bench_400", "llmbar_200"])
    ap.add_argument("--model", default="qwen2.5:7b-instruct", help='Ollama model, or "mock"')
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (for pilots)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    mode, prompt_name, norm_name = CONDITIONS[args.condition]
    template, norm = PROMPTS.get(prompt_name), NORMALISERS[norm_name]
    judge = MockJudge() if args.model == "mock" else OllamaJudge(args.model, num_ctx=args.num_ctx)

    rows = [json.loads(l) for l in open(f"data/{args.dataset}.jsonl")]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.results, exist_ok=True)
    tag = args.model.replace(":", "-").replace("/", "-")
    out_path = os.path.join(args.results, f"{tag}__{args.condition}__{args.dataset}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.condition} on {args.dataset} with {args.model}: {len(done)} done, {len(todo)} to go -> {out_path}")

    t0, warned = time.time(), False
    with open(out_path, "a") as f:
        for k, r in enumerate(todo, 1):
            verdict, outs = judge_item(judge, mode, template, r["question"],
                                       norm(r["response_a"]), norm(r["response_b"]))
            if not warned and any((o["prompt_tokens"] or 0) >= args.num_ctx - 50 for o in outs):
                print(f"  WARNING: prompt filled the {args.num_ctx}-token context; raise --num-ctx")
                warned = True
            f.write(json.dumps({
                "id": r["id"], "dataset": args.dataset, "condition": args.condition,
                "model": args.model, "verdict": verdict, "human": r.get("human_preference"),
                "bias_type": r.get("bias_type"), "category": r.get("metadata", {}).get("category"),
                "calls": len(outs),
                "prompt_tokens": sum(o["prompt_tokens"] or 0 for o in outs),
                "output_tokens": sum(o["output_tokens"] or 0 for o in outs),
                "seconds": round(sum(o["seconds"] for o in outs), 2),
                "raw": [o["text"][:1500] for o in outs]}) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"  {k}/{len(todo)}  {rate:.1f}s/item  ~{rate * (len(todo) - k) / 60:.0f} min left")


if __name__ == "__main__":
    main()
