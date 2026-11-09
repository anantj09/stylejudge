# Does Format Normalisation Fix Style Bias in Small LLM Judges?

An empirical evaluation of style bias, position bias, format normalisation, and pointwise scoring in small open-weight LLM-as-a-judge pipelines.

Built as a systematic follow-up to:
> Soumik (2026), *"Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines"*, [arXiv:2604.23178](https://arxiv.org/abs/2604.23178).

---

## 📌 Executive Summary & Key Findings

Soumik (2026) evaluated large proprietary API judges (e.g. GPT-4, Claude) and reached three main conclusions:
1. **Style bias** (preferring markdown over plain prose) is the dominant bias.
2. **Position bias** is negligible ($\le 0.04$).
3. **Recommendation**: Normalise response formatting before judging.

However, the paper **never empirically tested** its format normalisation recommendation, and evaluated **no small open-weight judges**. We tested these claims locally on consumer hardware (16 GB Mac via Ollama) across two 7B/8B models (`Qwen2.5-7B-Instruct` and `Llama-3.1-8B`):

* **Position bias dominates small judges**: Under baseline pairwise judging (B0), position effect reaches **+0.94 for Llama-3.1-8B** (47 of 50 pairs decided entirely by slot assignment) and **+0.61 for Qwen2.5-7B-Instruct** (32 of 50 pairs decided by slot). This directly contrasts with the $\le 0.04$ reported for frontier API models.
* **Position bias conceals style bias**: On Llama-3.1-8B, style bias appears near zero under B0 (+0.02) because extreme position bias masks it. Under pointwise evaluation (PW), true style bias emerges at **+0.22 [+0.09, +0.34]**.
* **Stripping markdown does not eliminate style bias**: Removing markdown syntax (headers, bold/italic, bullet markers, fences) does not remove the preference: Qwen's style bias shifts from **+0.19 to +0.25**, and MT-Bench markdown preference shifts from **+0.43 to +0.44** (vs. human preference of +0.16). Stripping removes surface tokens but leaves structural layout (e.g., list density, length) intact.
* **Pointwise judging eliminates position bias and curbs markdown over-reward**: Scoring each candidate answer independently (PW) completely eliminates slot bias by design and significantly reduces markdown over-reward relative to humans on MT-Bench mismatch pairs (**excess over humans drops from +0.26 to +0.07; $\Delta = -0.19$ [$-0.33, -0.06$]**), while maintaining overall human agreement (**57.8% vs. 57.5%**, McNemar $p = 1.000$).

---

## 📂 Repository Structure

```text
stylejudge/
├── .gitignore              # Ignores venvs, cache files, and temp clones
├── requirements.txt        # Minimal Python dependencies (numpy, requests)
├── README.md               # Project documentation and replication guide
├── prepare_data.py         # Downloads, samples, and verifies data against upstream repo
├── normalize.py            # Format normalisation routines (markdown strip & keep_code)
├── judge.py                # Prompt templates (NAIVE, COT, POINTWISE) and Ollama API client
├── run.py                  # Evaluation harness (pair, position-swap, pointwise modes)
├── analyze.py              # Statistical suite (bootstrap CIs, Cohen's kappa, McNemar test)
├── data/
│   ├── PROVENANCE.md       # Upstream commit hash and SHA-256 integrity checksums
│   ├── style_100.jsonl     # 50 controlled style pairs + 50 position-mirrored (style_ba) pairs
│   ├── mt_bench_400.jsonl  # 400 sampled MT-Bench human-judged pairs (Zheng et al., 2023)
│   └── llmbar_200.jsonl    # 200 sampled LLMBar pairs (Zeng et al., 2024)
└── results/
    ├── summary.md          # Generated benchmark tables and statistical comparisons
    └── *.jsonl             # Per-sample execution logs, prompt/output tokens, and verdicts
```

---

## 🧪 Experimental Conditions

| Condition | Mode | Judge Calls / Pair | Prompt | Formatting | Description |
|---|:---:|:---:|:---:|:---:|---|
| **B0** | `pair` | 1 | `NAIVE` | Raw | Paper baseline (single pairwise prompt) |
| **S1** | `swap` | 2 | `NAIVE` | Raw | Position-swap (both orders; agreement $\to$ verdict, disagreement $\to$ tie) |
| **STRIP** | `pair` | 1 | `NAIVE` | Stripped | The paper's untested recommendation (markdown stripped from both answers) |
| **PW** | `point` | 2 | `POINTWISE` | Raw | **Ours**: Score each answer independently (1–10); higher score wins, tie if equal |
| **PW_STRIP** | `point` | 2 | `POINTWISE` | Stripped | **Ours**: Pointwise evaluation on markdown-stripped answers |
| *IGNORE* | `pair` | 1 | `IGNORE_FORMAT` | Raw | Naive prompt + explicit "ignore formatting" clause (implemented; not in final run) |
| *S5* | `pair` | 1 | `COT` | Raw | Chain-of-thought prompt forcing reasoning steps (implemented) |
| *KEEPCODE* | `pair` | 1 | `NAIVE` | Kept Code | Markdown stripped, but code blocks preserved intact |
| *S5_STRIP* | `pair` | 1 | `COT` | Stripped | Chain-of-thought prompt with stripped formatting |

---

## 📊 Summary of Experimental Results

### 1. Controlled Style Pairs (`style_100`, $n=100$)
50 unique content pairs evaluated in both orientations (`style`: markdown in A; `style_ba`: markdown in B). Unbiased = 0.00 (correct answer is `tie`).

| Model | Condition | Style Bias (95% CI) | md in A | md in B | Tie Rate | Position Effect | Flips with Position |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Llama-3.1-8B** | B0 | +0.02 [−0.18, +0.20] | +0.96 | −0.92 | 0% | **+0.94** | 47 / 50 |
| | S1 | +0.02 [−0.03, +0.07] | +0.02 | +0.02 | 94% | +0.00 | 0 / 50 |
| | STRIP | +0.08 [−0.10, +0.26] | +1.00 | −0.84 | 0% | +0.92 | 46 / 50 |
| | **PW** | **+0.22 [+0.09, +0.34]** | +0.22 | +0.22 | 54% | **+0.00** | 0 / 50 |
| | PW_STRIP | +0.16 [+0.07, +0.26] | +0.16 | +0.16 | 76% | +0.00 | 0 / 50 |
| **Qwen2.5-7B-Instruct** | B0 | +0.19 [+0.00, +0.38] | +0.80 | −0.42 | 1% | **+0.61** | 32 / 50 |
| | S1 | +0.18 [+0.07, +0.29] | +0.18 | +0.18 | 66% | +0.00 | 0 / 50 |
| | STRIP | +0.25 [+0.07, +0.45] | +0.90 | −0.40 | 7% | +0.65 | 31 / 50 |
| | **PW** | **+0.00 [−0.10, +0.11]** | +0.00 | +0.00 | 68% | **+0.00** | 0 / 50 |
| | PW_STRIP | +0.12 [+0.02, +0.23] | +0.12 | +0.12 | 68% | +0.00 | 0 / 50 |

### 2. Real-World Benchmark (`mt_bench_400`, Qwen2.5-7B-Instruct)
Overall human agreement on 400 items, and markdown preference on the 129 "mismatch" pairs where only one answer uses markdown:

| Condition | Human Agreement | Cohen's $\kappa$ | $\Delta$ vs B0 ($p$-value) | Judge md-Pref | Human md-Pref | Excess over Humans | $\Delta$ Excess vs B0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | 57.5% [52.8, 62.3] | 0.314 | — | +0.43 [+0.27, +0.57] | +0.16 | +0.26 [+0.10, +0.43] | — |
| **S1** | 58.0% [53.0, 63.0] | 0.363 | +0.5 pp ($p=0.906$) | +0.36 [+0.22, +0.50] | +0.16 | +0.19 [+0.06, +0.33] | −0.07 [−0.16, +0.02] |
| **STRIP** | 57.2% [52.5, 62.0] | 0.310 | −0.2 pp ($p=1.000$) | +0.44 [+0.29, +0.58] | +0.16 | +0.28 [+0.12, +0.43] | +0.02 [+0.00, +0.05] |
| **PW** | 57.8% [53.0, 62.7] | 0.361 | +0.2 pp ($p=1.000$) | +0.23 [+0.09, +0.38] | +0.16 | **+0.07 [−0.07, +0.21]** | **−0.19 [−0.33, −0.06]** |
| **PW_STRIP** | 56.8% [51.5, 61.3] | 0.345 | −0.8 pp ($p=0.855$) | +0.22 [+0.07, +0.37] | +0.16 | **+0.06 [−0.09, +0.20]** | **−0.20 [−0.35, −0.06]** |

*Note: 95% confidence intervals are computed using 2,000 bootstrap resamples. Significance for $\Delta$ vs B0 agreement is evaluated via two-sided exact McNemar test on discordant pairs.*

---

## 🚀 Setup & Replication

### 1. Environment & Dependencies
```bash
git clone <your-repo-url>
cd stylejudge
pip install -r requirements.txt
```

### 2. Ollama Local LLM Setup
Ensure [Ollama](https://ollama.ai) is running locally on `http://localhost:11434`:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
```

### 3. Verify Data Provenance
Verify that all data files and baseline prompts match the upstream paper's repository:
```bash
python prepare_data.py --verify
```
*Expected output ends with `ALL CHECKS PASSED`.*

### 4. Running Experiments
To run offline sanity checks with the mock judge:
```bash
python run.py --condition B0 --dataset style_100 --model mock --limit 5
```

To run real evaluations (resumable; interrupts can be restarted without loss):
```bash
# Baseline pairwise evaluation
python run.py --condition B0 --dataset style_100 --model qwen2.5:7b-instruct
python run.py --condition B0 --dataset mt_bench_400 --model qwen2.5:7b-instruct

# Pointwise evaluation (Ours)
python run.py --condition PW --dataset style_100 --model qwen2.5:7b-instruct
python run.py --condition PW --dataset mt_bench_400 --model qwen2.5:7b-instruct
```

### 5. Generate Summary Tables
```bash
python analyze.py
```
This inspects all `.jsonl` files in `results/` and outputs statistical summaries to `results/summary.md`.

---

## 📖 Citation & Acknowledgments

This research reproduces and extends the experimental suite from:
```bibtex
@article{soumik2026judging,
  title={Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines},
  author={Soumik, S. K.},
  journal={arXiv preprint arXiv:2604.23178},
  year={2026}
}
```
All data and prompts from upstream are tracked and pinned at commit `6ff578c28366964af8e0558dba35930b1bd8a622` (MIT License). See [data/PROVENANCE.md](data/PROVENANCE.md) for full cryptographic verification.
