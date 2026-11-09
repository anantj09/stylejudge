# Data provenance

Every file in `data/` is rebuilt by `prepare_data.py` from the paper's official repository (https://github.com/sksoumik/judging-the-judges.git, MIT licence) at commit `6ff578c28366964af8e0558dba35930b1bd8a622`, with the paper's own sampling (`numpy.random.default_rng(42)`, as in its `src/data_loader.py`).

| file | path in repo | original dataset | rows | SHA-256 |
|---|---|---|---|---|
| style_100.jsonl | `data/custom/controlled_pairs.jsonl` | controlled STYLE + STYLE-BA pairs built by the paper's author | 100 | `253372fd1746addbc33141169cf508f5e03b460881ff74549afd4dada25d7fe7` |
| mt_bench_400.jsonl | `data/raw/mt_bench/mt_bench.jsonl` | MT-Bench human judgments (Zheng et al., 2023), lmsys/mt_bench_human_judgments | 400 | `e3592636da0853b31564dfae4ac06f6a942e6256f2ba121f3320c49fb4c14d79` |
| llmbar_200.jsonl | `data/raw/llmbar/llmbar.jsonl` | LLMBar (Zeng et al., 2024), princeton-nlp/LLMBar | 200 | `43cf319fb3476f0cc297ed1e7d94d050ff07dc1893a88e7f5a6021f0978aa749` |

Verify at any time with `python prepare_data.py --verify`.
