"""Pairwise judge on a local Ollama model, using the paper's prompts verbatim.

Uses Ollama's native /api/chat (not the OpenAI-compatible endpoint) so we can raise the
context window: the default 4096 tokens would silently truncate long MT-Bench pairs.
"""
import json
import re
import time

import requests

# --- prompts copied verbatim from the paper's repo (src/judges/prompts.py) ---
NAIVE = """\
You are an expert evaluator. Given a question and two responses, determine which response is better.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Evaluate both responses and provide your verdict. Output a JSON object:
{{
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>,
  "reasoning": "<brief explanation>"
}}"""

COT = """\
You are an expert evaluator. Analyze two responses step by step before making a judgment.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Follow these steps EXACTLY:

Step 1: Identify the key requirements of the question.
Step 2: Evaluate Response A's strengths and weaknesses against those requirements.
Step 3: Evaluate Response B's strengths and weaknesses against those requirements.
Step 4: Compare the two responses directly on substance (ignore length and formatting).
Step 5: Provide your final verdict.

Output JSON:
{{
  "step1_requirements": "<key requirements>",
  "step2_analysis_a": "<Response A analysis>",
  "step3_analysis_b": "<Response B analysis>",
  "step4_comparison": "<direct comparison>",
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>
}}"""

# --- the other half of the paper's recommendation: tell the judge to ignore formatting ---
IGNORE_FORMAT = NAIVE.replace(
    "Evaluate both responses and provide your verdict.",
    "Ignore formatting entirely (markdown, headers, bullet points, bold text, tables): "
    "judge only the substance of what each response says.\n\n"
    "Evaluate both responses and provide your verdict.")

# --- pointwise (ours): the naive prompt adapted to ONE response, so there are no A/B slots ---
POINTWISE = """\
You are an expert evaluator. Given a question and a response, rate how good the response is.

Question: {question}

Response: {response}

Evaluate the response and provide your rating. Output a JSON object:
{{
  "score": <float 1-10>,
  "reasoning": "<brief explanation>"
}}"""

PROMPTS = {"naive": NAIVE, "cot": COT, "ignore_format": IGNORE_FORMAT}

SCORE_ONLY_RE = re.compile(r'"score"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)')


def parse_score(text):
    """Return the 1-10 score from a pointwise judgement, or None."""
    try:
        v = float(json.loads(text).get("score"))
        return v if 0 <= v <= 10 else None
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        m = SCORE_ONLY_RE.search(text or "")
        return float(m.group(1)) if m else None

VERDICT_RE = re.compile(r'"verdict"\s*:\s*"?\s*(A|B|tie|Tie|TIE)\b')
SCORE_RE = {k: re.compile(rf'"{k}"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)') for k in ("score_a", "score_b")}


def parse_verdict(text):
    """Return 'A', 'B', 'tie', or None. Falls back to comparing scores if no verdict field."""
    try:
        obj = json.loads(text)
        v = str(obj.get("verdict", "")).strip()
        if v.upper() in ("A", "B"):
            return v.upper()
        if v.lower() == "tie":
            return "tie"
    except (json.JSONDecodeError, AttributeError):
        pass
    m = VERDICT_RE.search(text or "")
    if m:
        v = m.group(1)
        return "tie" if v.lower() == "tie" else v
    sa, sb = (SCORE_RE[k].search(text or "") for k in ("score_a", "score_b"))
    if sa and sb:
        a, b = float(sa.group(1)), float(sb.group(1))
        return "A" if a > b else "B" if b > a else "tie"
    return None


class OllamaJudge:
    def __init__(self, model, host="http://localhost:11434", num_ctx=8192,
                 temperature=0.1, max_tokens=700, seed=0):
        self.model, self.host = model, host
        self.options = {"num_ctx": num_ctx, "temperature": temperature,
                        "num_predict": max_tokens, "seed": seed}

    def __call__(self, prompt):
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", timeout=600, json={
            "model": self.model, "stream": False, "format": "json",
            "messages": [{"role": "user", "content": prompt}], "options": self.options})
        r.raise_for_status()
        d = r.json()
        return {"text": d["message"]["content"],
                "prompt_tokens": d.get("prompt_eval_count"),
                "output_tokens": d.get("eval_count"),
                "seconds": round(time.time() - t0, 2)}


class MockJudge:
    """Offline stand-in for testing the pipeline: prefers whichever side has more markdown.
    Never report its numbers."""
    model = "mock"

    def __call__(self, prompt):
        score = lambda t: t.count("**") + t.count("\n- ") + t.count("#")
        if "Response A:" not in prompt:                      # pointwise prompt
            r = prompt.split("Response:", 1)[1]
            return {"text": json.dumps({"score": min(10, 5 + score(r)), "reasoning": "mock"}),
                    "prompt_tokens": len(prompt) // 4, "output_tokens": 20, "seconds": 0.0}
        a = prompt.split("Response A:", 1)[1].split("Response B:", 1)[0]
        b = prompt.split("Response B:", 1)[1]
        score = lambda t: t.count("**") + t.count("\n- ") + t.count("#")
        v = "A" if score(a) > score(b) else "B" if score(b) > score(a) else "tie"
        return {"text": json.dumps({"verdict": v, "score_a": 5, "score_b": 5, "reasoning": "mock"}),
                "prompt_tokens": len(prompt) // 4, "output_tokens": 20, "seconds": 0.0}
