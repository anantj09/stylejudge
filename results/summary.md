
## llama3.1:8b — style_100

| condition | calls/pair | n | style bias (95% CI) | md in A | md in B | tie rate | invalid |
|---|---|---|---|---|---|---|---|
| B0 | 1 | 100 | +0.02 [-0.18, +0.20] | +0.96 | -0.92 | 0% | 0% |
| S1 | 2 | 100 | +0.02 [-0.03, +0.07] | +0.02 | +0.02 | 94% | 0% |
| STRIP | 1 | 100 | +0.08 [-0.10, +0.26] | +1.00 | -0.84 | 0% | 0% |
| PW | 2 | 100 | +0.22 [+0.09, +0.34] | +0.22 | +0.22 | 54% | 0% |
| PW_STRIP | 2 | 100 | +0.16 [+0.07, +0.26] | +0.16 | +0.16 | 76% | 0% |

Unbiased judge = 0. Positive = prefers markdown. 'md in A/B' split shows position effects.

### Same pair, both orders (position-consistency)

| condition | pairs | md both times | prose both times | flips with position | involves a tie | position effect | style effect (swap-consistent) |
|---|---|---|---|---|---|---|---|
| B0 | 50 | 2 | 1 | 47 | 0 | +0.94 | +0.02 |
| S1 | 50 | 2 | 1 | 0 | 47 | +0.00 | +0.02 |
| STRIP | 50 | 4 | 0 | 46 | 0 | +0.92 | +0.08 |
| PW | 50 | 17 | 6 | 0 | 27 | +0.00 | +0.22 |
| PW_STRIP | 50 | 10 | 2 | 0 | 38 | +0.00 | +0.16 |

position effect = P(pick slot A) − P(pick slot B) over both orders (0 = none). style effect counts only pairs where the judge picked the same content in both orders, i.e. the paper's position-swap rule (S1) applied to data we already have.

## qwen2.5:7b-instruct — mt_bench_400

| condition | calls/pair | n | agreement (95% CI) | kappa | Δ vs B0 (95% CI) | McNemar p | tie rate | invalid |
|---|---|---|---|---|---|---|---|---|
| B0 | 1 | 400 | 57.5% [52.8%, 62.3%] | 0.314 | — | — | 1% | 0% |
| S1 | 2 | 400 | 58.0% [53.0%, 63.0%] | 0.363 | +0.5 pp [-3.5, +4.5] | 0.906 | 28% | 0% |
| STRIP | 1 | 400 | 57.2% [52.5%, 62.0%] | 0.310 | -0.2 pp [-1.0, +0.0] | 1.000 | 1% | 0% |
| PW | 2 | 400 | 57.8% [53.0%, 62.7%] | 0.361 | +0.2 pp [-5.2, +5.8] | 1.000 | 29% | 0% |
| PW_STRIP | 2 | 400 | 56.8% [51.5%, 61.3%] | 0.345 | -0.8 pp [-6.0, +4.8] | 0.855 | 28% | 0% |

Subsets: mismatch = exactly one side uses markdown (n=129); code = a fenced code block appears (n=57).

| condition | agree: mismatch | agree: rest | agree: code | md-side pref: judge (95% CI) | humans | excess over humans (95% CI) | Δ excess vs B0 (95% CI) |
|---|---|---|---|---|---|---|---|
| B0 | 60.5% | 56.1% | 70.2% | +0.43 [+0.27, +0.57] | +0.16 | +0.26 [+0.10, +0.43] | — |
| S1 | 55.0% | 59.4% | 71.9% | +0.36 [+0.22, +0.50] | +0.16 | +0.19 [+0.06, +0.33] | -0.07 [-0.16, +0.02] |
| STRIP | 60.5% | 55.7% | 68.4% | +0.44 [+0.29, +0.58] | +0.16 | +0.28 [+0.12, +0.43] | +0.02 [+0.00, +0.05] |
| PW | 58.9% | 57.2% | 68.4% | +0.23 [+0.09, +0.38] | +0.16 | +0.07 [-0.07, +0.21] | -0.19 [-0.33, -0.06] |
| PW_STRIP | 57.4% | 56.5% | 64.9% | +0.22 [+0.07, +0.37] | +0.16 | +0.06 [-0.09, +0.20] | -0.20 [-0.35, -0.06] |

md-side pref = P(pick markdown side) − P(pick plain side) on mismatch pairs. If the judge's value is far above the humans', it over-rewards markdown. Δ excess vs B0 < 0 with a CI excluding 0 means the method significantly reduces that over-reward.

## qwen2.5:7b-instruct — style_100

| condition | calls/pair | n | style bias (95% CI) | md in A | md in B | tie rate | invalid |
|---|---|---|---|---|---|---|---|
| B0 | 1 | 100 | +0.19 [+0.00, +0.38] | +0.80 | -0.42 | 1% | 0% |
| S1 | 2 | 100 | +0.18 [+0.07, +0.29] | +0.18 | +0.18 | 66% | 0% |
| STRIP | 1 | 100 | +0.25 [+0.07, +0.45] | +0.90 | -0.40 | 7% | 0% |
| PW | 2 | 100 | +0.00 [-0.10, +0.11] | +0.00 | +0.00 | 68% | 0% |
| PW_STRIP | 2 | 100 | +0.12 [+0.02, +0.23] | +0.12 | +0.12 | 68% | 0% |

Unbiased judge = 0. Positive = prefers markdown. 'md in A/B' split shows position effects.

### Same pair, both orders (position-consistency)

| condition | pairs | md both times | prose both times | flips with position | involves a tie | position effect | style effect (swap-consistent) |
|---|---|---|---|---|---|---|---|
| B0 | 50 | 13 | 4 | 32 | 1 | +0.61 | +0.18 |
| S1 | 50 | 13 | 4 | 0 | 33 | +0.00 | +0.18 |
| STRIP | 50 | 13 | 1 | 31 | 5 | +0.65 | +0.24 |
| PW | 50 | 8 | 8 | 0 | 34 | +0.00 | +0.00 |
| PW_STRIP | 50 | 11 | 5 | 0 | 34 | +0.00 | +0.12 |

position effect = P(pick slot A) − P(pick slot B) over both orders (0 = none). style effect counts only pairs where the judge picked the same content in both orders, i.e. the paper's position-swap rule (S1) applied to data we already have.
