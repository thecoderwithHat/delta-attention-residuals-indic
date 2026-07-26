"""Inference and structural scoring for NVIDIA's BFCL-Hi benchmark."""

import ast
import json
import math
import re
from collections import defaultdict


DATASET_REPO = "nvidia/BFCL-Hi"
DEFAULT_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "relevance",
    "irrelevance",
)
ANSWER_CATEGORIES = frozenset(
    {"simple", "multiple", "parallel", "parallel_multiple"}
)


def _load_json_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_category(category):
    """Download and join one BFCL-Hi category with its published answers."""
    if category not in DEFAULT_CATEGORIES:
        raise ValueError(
            f"Unknown BFCL-Hi category {category!r}; "
            f"choose from {', '.join(DEFAULT_CATEGORIES)}"
        )

    from huggingface_hub import hf_hub_download

    filename = f"BFCL_v2_live_{category}.json"
    data_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
    )
    records = _load_json_lines(data_path)

    if category in ANSWER_CATEGORIES:
        answer_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"possible_answer/{filename}",
            repo_type="dataset",
        )
        answers = {
            row["id"]: row["ground_truth"] for row in _load_json_lines(answer_path)
        }
        missing = [row["id"] for row in records if row["id"] not in answers]
        if missing:
            raise ValueError(
                f"Missing BFCL-Hi ground truth for {len(missing)} {category} rows"
            )
        for row in records:
            row["ground_truth"] = answers[row["id"]]

    return records


def _parse_jsonish(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return value


def _as_call(value):
    value = _parse_jsonish(value)
    if not isinstance(value, dict):
        return None

    if value.get("type") == "function" and isinstance(value.get("function"), dict):
        value = value["function"]
    if "function" in value and isinstance(value["function"], dict):
        value = value["function"]

    if "name" in value:
        arguments = value.get("arguments", value.get("parameters", {}))
        arguments = _parse_jsonish(arguments)
        if isinstance(arguments, dict):
            return {"name": str(value["name"]), "arguments": arguments}

    if len(value) == 1:
        name, arguments = next(iter(value.items()))
        arguments = _parse_jsonish(arguments)
        if isinstance(arguments, dict):
            return {"name": str(name), "arguments": arguments}
    return None


def parse_tool_calls(response):
    """Extract OpenAI, Qwen, or BFCL-style function calls from model text."""
    tagged = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL
    )
    candidates = tagged

    if not candidates:
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        cleaned = re.sub(r"```(?:json|python)?", "", cleaned).replace("```", "")
        candidates = [cleaned.strip()]

    calls = []
    for candidate in candidates:
        parsed = _parse_jsonish(candidate)
        values = parsed if isinstance(parsed, list) else [parsed]
        for value in values:
            call = _as_call(value)
            if call is not None:
                calls.append(call)

    return calls


def _values_equal(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual:
                if isinstance(value, list) and "" in value:
                    continue
                return False
            if not _matches_spec(actual[key], value):
                return False
        allowed_extra = set(actual) - set(expected)
        return not allowed_extra

    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _matches_spec(got, want) for got, want in zip(actual, expected)
        )

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def _matches_spec(actual, alternatives):
    if not isinstance(alternatives, list):
        return _values_equal(actual, alternatives)
    if not alternatives:
        return actual == []
    return any(_values_equal(actual, candidate) for candidate in alternatives)


def _call_matches(predicted, expected):
    if len(expected) != 1:
        return False
    name, argument_spec = next(iter(expected.items()))
    return (
        predicted["name"] == name
        and isinstance(predicted["arguments"], dict)
        and _values_equal(predicted["arguments"], argument_spec)
    )


def calls_match(predicted, expected):
    """Match calls one-to-one, allowing parallel calls in any order."""
    if len(predicted) != len(expected):
        return False
    unmatched = list(expected)
    for call in predicted:
        for index, target in enumerate(unmatched):
            if _call_matches(call, target):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def score_response(category, response, ground_truth=None):
    calls = parse_tool_calls(response)
    if category == "relevance":
        return bool(calls), bool(calls)
    if category == "irrelevance":
        return not calls, True
    return calls_match(calls, ground_truth or []), bool(calls)


def _tool_schemas(functions):
    return [
        {
            "type": "function",
            "function": {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            },
        }
        for function in functions
    ]


def generate_response(
    model,
    tokenizer,
    row,
    device,
    max_new_tokens=1024,
):
    """Generate one tool-aware response using the model's chat template."""
    import torch

    tools = _tool_schemas(row["function"])
    try:
        prompt = tokenizer.apply_chat_template(
            row["question"],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "BFCL-Hi requires a tokenizer chat template with tool support"
        ) from exc

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=False)


def evaluate_bfcl_hi(
    model,
    tokenizer,
    device,
    categories=DEFAULT_CATEGORIES,
    limit=None,
    max_new_tokens=1024,
):
    """Evaluate BFCL-Hi and return aggregate and per-category metrics."""
    totals = defaultdict(int)
    category_results = {}

    for category in categories:
        records = load_category(category)
        if limit is not None:
            records = records[:limit]

        correct = 0
        parsed = 0
        for index, row in enumerate(records, start=1):
            response = generate_response(
                model,
                tokenizer,
                row,
                device,
                max_new_tokens=max_new_tokens,
            )
            is_correct, is_parsed = score_response(
                category,
                response,
                row.get("ground_truth"),
            )
            correct += int(is_correct)
            parsed += int(is_parsed)
            print(
                f"\r  BFCL-Hi/{category}: {index}/{len(records)}",
                end="",
                flush=True,
            )
        print()

        count = len(records)
        accuracy = correct / count if count else 0.0
        parse_rate = parsed / count if count else 0.0
        category_results[category] = {
            "accuracy": accuracy,
            "parse_rate": parse_rate,
            "samples": count,
        }
        totals["correct"] += correct
        totals["parsed"] += parsed
        totals["samples"] += count

    sample_count = totals["samples"]
    return {
        "accuracy": totals["correct"] / sample_count if sample_count else 0.0,
        "parse_rate": totals["parsed"] / sample_count if sample_count else 0.0,
        "samples": sample_count,
        "categories": category_results,
    }
