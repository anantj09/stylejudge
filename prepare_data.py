"""Rebuild data/ from the paper's official repository and verify everything.

  python prepare_data.py            # clone repo at the pinned commit, rebuild data/, write data/PROVENANCE.md
  python prepare_data.py --verify   # rebuild into a temp folder; check bundled data and prompts match

Source: github.com/sksoumik/judging-the-judges (MIT licence), the official code and data release
of arXiv 2604.23178. Sampling is identical to the paper's src/data_loader.py:
numpy default_rng(seed=42).choice(N, size, replace=False), indices sorted.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO_URL = "https://github.com/sksoumik/judging-the-judges.git"
COMMIT = "6ff578c28366964af8e0558dba35930b1bd8a622"   # 24 Jun 2026

# output file: (path inside the repo, sample size or "style")
SOURCES = {
    "style_100.jsonl": ("data/custom/controlled_pairs.jsonl", "style"),
    "mt_bench_400.jsonl": ("data/raw/mt_bench/mt_bench.jsonl", 400),
    "llmbar_200.jsonl": ("data/raw/llmbar/llmbar.jsonl", 200),
}
UPSTREAM = {
    "style_100.jsonl": "controlled STYLE + STYLE-BA pairs built by the paper's author",
    "mt_bench_400.jsonl": "MT-Bench human judgments (Zheng et al., 2023), lmsys/mt_bench_human_judgments",
    "llmbar_200.jsonl": "LLMBar (Zeng et al., 2024), princeton-nlp/LLMBar",
}


def git(*args, cwd=None, out=False):
    r = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=out, text=True)
    return r.stdout.strip() if out else None


def get_repo(repo_dir):
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        git("clone", REPO_URL, repo_dir)
    git("checkout", "--quiet", COMMIT, cwd=repo_dir)
    head = git("rev-parse", "HEAD", cwd=repo_dir, out=True)
    if head != COMMIT:
        sys.exit(f"repo is at {head}, expected {COMMIT}")
    return repo_dir


def sample(rows, n, seed=42):
    if n >= len(rows):
        return rows
    idx = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(idx)]


def build(repo_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, (src, how) in SOURCES.items():
        rows = [json.loads(l) for l in open(os.path.join(repo_dir, src))]
        if how == "style":
            rows = [r for r in rows if r["bias_type"] in ("style", "style_ba")]
        else:
            rows = sample(rows, how)
        with open(os.path.join(out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_prompts(repo_dir):
    """Our NAIVE and COT prompts must be byte-identical to the paper's."""
    path = os.path.join(repo_dir, "src", "judges", "prompts.py")
    spec = importlib.util.spec_from_file_location("paper_prompts", path)
    theirs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theirs)
    import judge as ours
    ok = True
    for mine, their_name in [("NAIVE", "NAIVE_JUDGE_PROMPT"), ("COT", "COT_FORCING_PROMPT")]:
        same = getattr(ours, mine) == getattr(theirs, their_name)
        ok &= same
        print(f"  prompt {mine:5s} vs paper's {their_name}: {'IDENTICAL' if same else 'DIFFERENT'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="judging-the-judges")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    repo = get_repo(args.repo_dir)
    print(f"Paper repository: {REPO_URL} at commit {COMMIT[:7]}")

    if args.verify:
        ok = True
        with tempfile.TemporaryDirectory() as tmp:
            build(repo, tmp)
            for name in SOURCES:
                same = sha256(os.path.join(tmp, name)) == sha256(os.path.join("data", name))
                ok &= same
                print(f"  data/{name}: {'matches rebuild from repo' if same else 'DIFFERS from repo'}")
        ok &= check_prompts(repo)
        print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
        sys.exit(0 if ok else 1)

    build(repo, "data")
    lines = ["# Data provenance", "",
             f"Every file in `data/` is rebuilt by `prepare_data.py` from the paper's official repository "
             f"({REPO_URL}, MIT licence) at commit `{COMMIT}`, with the paper's own sampling "
             "(`numpy.random.default_rng(42)`, as in its `src/data_loader.py`).", "",
             "| file | path in repo | original dataset | rows | SHA-256 |", "|---|---|---|---|---|"]
    for name, (src, _) in SOURCES.items():
        p = os.path.join("data", name)
        lines.append(f"| {name} | `{src}` | {UPSTREAM[name]} | {sum(1 for _ in open(p))} | `{sha256(p)}` |")
    lines += ["", "Verify at any time with `python prepare_data.py --verify`."]
    open("data/PROVENANCE.md", "w").write("\n".join(lines) + "\n")
    print("Rebuilt data/ and wrote data/PROVENANCE.md")
    check_prompts(repo)


if __name__ == "__main__":
    main()
