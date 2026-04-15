"""
q1.py  –  Prompt Engineering: Extracting Per-Session Progress Scores

Pipeline overview
-----------------
    data/labeled_notes.json   ──► score ──► compute_metrics() ──► print results  (Q1a: validate prompt)
    data/unlabeled_notes.json ──► score ──► save                                  (Q1b: score at scale)

The LLM's job
-------------
For each client, the model receives the full sequence of session notes and
must return one progress score per consecutive note pair:

    notes 1→2 : score
    notes 2→3 : score
    ...
    notes 11→12 : score

Scores are integers 0–3, returned as a JSON list, e.g. [2, 1, 0, 0, 1, ...].

What is already done for you
------------------------------
- Parsing and validating the LLM's JSON response
- Retrying once automatically if the response is malformed
- Looping over every client in a dataset
- Aligning true vs. predicted scores into a flat list of (true, predicted) pairs
- Building and printing the confusion matrix
- Saving all outputs to JSON

Your tasks  (search for # TODO to find each one)
--------------------------------------------------
1. build_prompt()      Write the prompt that instructs the LLM.
2. call_llm()          Wire up your chosen LLM API (OpenAI, Gemini, Anthropic, etc.).
3. compute_metrics()   Define and compute the performance metric(s) you will use
                       to evaluate and compare prompt versions.

Expected inputs:
    data/labeled_notes.json     – hand-scored by Patel; use this to test your prompt
    data/unlabeled_notes.json   – apply your validated prompt here

Expected outputs:
    output/q1/evaluated_labeled_results.json   – scored test set with true labels (Q1a)
    output/q1/scored_notes.json                – scored unlabeled clients (Q1b, feeds Q2)
    output/q1/scored_notes.csv                 – Q1b CSV: client_id, session, score
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class BaseQ1Config:
    client_id_key: str = "client_id"
    notes_key: str = "notes"
    note_number_key: str = "note_number"
    note_text_key: str = "note_text"
    true_vector_key: str = "scored_progress"
    pred_vector_key: str = "estimated_trajectory_vector"

    # Fixed: ground-truth uses 0–3 scale, not 1–4
    valid_scores: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class Q1ALabeledConfig(BaseQ1Config):
    test_path: str = "data/labeled_notes.json"
    evaluated_output_path: str = "output/q1/evaluated_labeled_results.json"


@dataclass
class Q1BUnlabeledConfig(BaseQ1Config):
    unlabeled_path: str = "data/unlabeled_notes.json"
    output_path: str = "output/q1/scored_notes.json"


# ============================================================================
# DATA LOADING / SAVING
# ============================================================================

def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent folders for an output path and return it as a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a top-level JSON list from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}.")
    return data


def save_json(data: Any, path: str) -> None:
    """Save JSON to disk and create parent folders if needed."""
    output_path = ensure_parent_dir(path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def get_vector_pair(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> tuple[str, List[int], List[int]]:
    """Pull the client id, true vector, and estimated vector from one scored record."""
    client_id = str(record[config.client_id_key])
    true_vector = record.get(config.true_vector_key, [])
    estimated_vector = record.get(config.pred_vector_key, [])
    return client_id, true_vector, estimated_vector


def build_step_comparisons(
    client_id: str,
    true_vector: List[int],
    estimated_vector: List[int],
) -> List[Dict[str, Any]]:
    """Build one row per compared step between the true and estimated vectors."""
    rows = []
    for step_idx, (true_score, estimated_score) in enumerate(
        zip(true_vector, estimated_vector),
        start=1,
    ):
        rows.append(
            {
                "client_id": client_id,
                "step_number": step_idx,
                "true_score": true_score,
                "estimated_score": estimated_score,
            }
        )
    return rows


def build_client_comparison(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Create the per-client comparison payload used by evaluation code."""
    client_id, true_vector, estimated_vector = get_vector_pair(record, config)
    step_rows = build_step_comparisons(client_id, true_vector, estimated_vector)
    return {
        "client_id": client_id,
        "true_vector": true_vector,
        "estimated_vector": estimated_vector,
        "n_true_scores": len(true_vector),
        "n_estimated_scores": len(estimated_vector),
        "n_compared_scores": len(step_rows),
        "step_comparisons": step_rows,
    }


def build_evaluation_comparisons(
    scored_test_data: List[Dict[str, Any]],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Build client-level and step-level comparison tables for evaluation."""
    client_level_comparisons = []
    step_level_comparisons = []

    for record in scored_test_data:
        client_summary = build_client_comparison(record, config)
        client_level_comparisons.append(client_summary)
        step_level_comparisons.extend(client_summary["step_comparisons"])

    return {
        "n_clients": len(scored_test_data),
        "client_level_comparisons": client_level_comparisons,
        "step_level_comparisons": step_level_comparisons,
    }


def build_confusion_matrix(
    step_rows: List[Dict[str, Any]],
    valid_scores: List[int] | tuple[int, ...],
) -> Dict[str, Any]:
    """Build a confusion matrix with row totals, column totals, and a printable table."""
    matrix = {
        true_score: {estimated_score: 0 for estimated_score in valid_scores}
        for true_score in valid_scores
    }

    for row in step_rows:
        true_score = row["true_score"]
        estimated_score = row["estimated_score"]
        if true_score in matrix and estimated_score in matrix[true_score]:
            matrix[true_score][estimated_score] += 1

    row_totals = {
        true_score: sum(
            matrix[true_score][estimated_score] for estimated_score in valid_scores
        )
        for true_score in valid_scores
    }
    column_totals = {
        estimated_score: sum(
            matrix[true_score][estimated_score] for true_score in valid_scores
        )
        for estimated_score in valid_scores
    }
    grand_total = sum(row_totals.values())

    headers = ["true\\pred", *[str(score) for score in valid_scores], "Total"]
    row_label_width = max(
        len(headers[0]),
        len("Total"),
        max(len(str(score)) for score in valid_scores),
    )
    cell_width = max(
        5,
        max(
            len(str(value))
            for value in [
                *[
                    matrix[true_score][estimated_score]
                    for true_score in valid_scores
                    for estimated_score in valid_scores
                ],
                *row_totals.values(),
                *column_totals.values(),
                grand_total,
            ]
        ),
    )

    header_line = " | ".join(
        [headers[0].rjust(row_label_width)]
        + [header.rjust(cell_width) for header in headers[1:]]
    )
    separator_line = "-+-".join(
        ["-" * row_label_width] + ["-" * cell_width for _ in headers[1:]]
    )

    table_lines = [header_line, separator_line]
    for true_score in valid_scores:
        row_values = [
            str(matrix[true_score][estimated_score])
            for estimated_score in valid_scores
        ]
        row_line = " | ".join(
            [str(true_score).rjust(row_label_width)]
            + [value.rjust(cell_width) for value in row_values]
            + [str(row_totals[true_score]).rjust(cell_width)]
        )
        table_lines.append(row_line)

    total_line = " | ".join(
        ["Total".rjust(row_label_width)]
        + [
            str(column_totals[estimated_score]).rjust(cell_width)
            for estimated_score in valid_scores
        ]
        + [str(grand_total).rjust(cell_width)]
    )
    table_lines.append(separator_line)
    table_lines.append(total_line)

    return {
        "labels": list(valid_scores),
        "counts": matrix,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "grand_total": grand_total,
        "table": "\n".join(table_lines),
    }


# ============================================================================
# TODO 1 of 3 — PROMPT
# ============================================================================

# System persona — included in the Gemini prompt as top-level instructions.
# Kept separate so the user message (returned by build_prompt) stays focused
# on the notes and scoring instructions.
_SYSTEM_PROMPT = (
    "You are David Patel, a senior speech-language pathologist (SLP) with 20 years "
    "of experience evaluating children's therapy progress. You specialize in reviewing "
    "sequential session notes and judging how much clinical progress occurred between "
    "consecutive sessions.\n\n"
    "Your evaluation approach:\n"
    "- You read notes IN ORDER and compare each note strictly to the one immediately before it.\n"
    "- You focus on: movement through the goal hierarchy, changes in cueing/support needs, "
    "consistency and accuracy of target productions, generalization to new contexts, and "
    "spontaneity of correct productions.\n"
    "- You are calibrated and conservative: score 0 (maintenance) is by far the most common "
    "outcome (~52% of transitions). Scores of 2 and 3 require clear, documented evidence. "
    "Score 3 is rare (~9%) and reserved for unmistakable hierarchy-level jumps."
)


def build_prompt(notes_json_str: str) -> str:
    """
    Build the user-turn prompt for scoring one client's full note sequence.

    Strategy (per assignment brief):
    - Explicit 0–3 score definitions with clinical anchor signals
    - Boundary clarifications for adjacent scores (0v1, 1v2, 2v3)
    - Reference to Exhibit A goal hierarchy and independence levels
    - Distribution reminder (0 is dominant; 3 is rare)
    - Output constraint: plain JSON list only (matches parse_vector_from_response)
    - No few-shot examples — definitions do the work

    Parameters
    ----------
    notes_json_str : str
        The client's full note sequence as a JSON string.
        Each note is a dict with keys "note_number" and "note_text".

    Returns
    -------
    str
        The complete user-turn prompt to send to the LLM.
    """
    notes = json.loads(notes_json_str)
    n_sessions = len(notes)
    n_transitions = n_sessions - 1
    transition_list = "\n".join(
        f"  Score {i+1}: Session {notes[i]['note_number']}→{notes[i+1]['note_number']}"
        for i in range(n_transitions)
    )

    # Format notes as readable text blocks
    notes_text = ""
    for note in notes:
        notes_text += f"\n--- Session {note['note_number']} ---\n{note['note_text']}\n"

    return f"""Below are {n_sessions} consecutive session notes for a pediatric speech-language therapy client. Your task is to assign a progress score for each consecutive pair of sessions.

=== RATING SCALE (0–3) ===

Score 0 — Maintenance / minimal change
The child is functioning at essentially the same level as the previous session. Accuracy, cueing needs, and hierarchy position remain similar. THIS IS THE MOST COMMON SCORE — most adjacent sessions show maintenance or consolidation.
Key signals: "similar to last session", "performance remained stable", "comparable to previous week", same cueing level needed, same accuracy range, clinician running the same activities at the same level with no noted change.

Score 1 — Small but clear improvement
Modest but concrete progress within the same general level — slightly better consistency, slightly less cueing, or improved carryover — WITHOUT a major jump in independence or hierarchy level.
Key signals: "slightly improved", "somewhat better", a small measurable gain (e.g., 5–15% accuracy increase within the same task), parent noticing a minor positive change, one target area shows a minor step forward while others remain stable.

Score 2 — Meaningful clinical progress
An obvious, clinically significant step forward that matters: moving from inconsistent to fairly consistent performance, clearly requiring less support, or showing broader generalization across new contexts.
Key signals: clinician explicitly notes "noticeably improved" or "significant progress", accuracy jumps meaningfully, cueing drops a full level (maximal→moderate or moderate→minimal), new contexts or settings mastered, multiple goals showing simultaneous improvement.

Score 3 — Major gain / step up in hierarchy level
A clear breakthrough: a goal-hierarchy jump, a major gain in independence, or a new level of spontaneous use. USE SPARINGLY — reserved for unmistakable leaps.
Key signals: moving up the goal hierarchy (isolation→syllable→word→carrier phrase→sentence→conversation), shifting from imitation to spontaneous production, first emergence of self-correction, dramatic reduction in support needs across the board.

=== BOUNDARY CLARIFICATIONS ===

0 vs 1: If there is ANY concrete improvement documented, even a small one, that is at least a 1. Reserve 0 for cases where the clinician describes no change or purely lateral movement.

1 vs 2: A score of 2 requires MULTIPLE clear improvements OR one substantial shift (e.g., cueing drops a full level). A single "slightly better" finding is a 1, not a 2.

2 vs 3: A score of 3 requires a hierarchy level change or a dramatic shift toward independence. Strong improvement that stays within the same hierarchy level is a 2, not a 3. Score 3 is rare but should not be avoided when the evidence clearly shows a hierarchy jump or dramatic independence shift.

=== GOAL HIERARCHY (Exhibit A) ===

Production hierarchy (lowest → highest):
  isolation → syllable → word → carrier phrase → sentence → conversational speech

Independence levels (most support → least support):
  imitation → maximal cueing → moderate cueing → minimal cueing → spontaneous

Progress means moving up either axis. A hierarchy jump between sessions = score 3.
Improvement within the same level = score 1 or 2 depending on magnitude.

=== SESSION NOTES ===
{notes_text}

=== YOUR TASK ===

Score each of the following {n_transitions} transitions in order:
{transition_list}

Your response must contain exactly {n_transitions} scores — one per line above, in order.

Return ONLY a bare JSON array with exactly {n_transitions} integers — no wrapper object, no "scores" key, no markdown fences, no explanation. Your entire response must be parseable by json.loads() as a list.

Correct format:
[1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0]

Incorrect formats (do NOT use):
{{"scores": [1, 0, 2, ...]}}
```json
[1, 0, 2, ...]
```"""


# ============================================================================
# TODO 2 of 3 — LLM CALL
# ============================================================================

def call_llm(prompt: str) -> str:
    """
    Send a prompt to Gemini via Google Generative Language API and return raw text.

    Uses:
    - GEMINI_API_KEY environment variable
    - GEMINI_MODEL environment variable (defaults to gemini-pro)
    - temperature=0.0 for deterministic scoring
    - maxOutputTokens=1024 to allow the model to return the JSON list

    Parameters
    ----------
    prompt : str
        The string returned by build_prompt().

    Returns
    -------
    str
        The model's raw text response (parsing happens in parse_vector_from_response).
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY in your environment or .env file."
        )

    model = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite")
    if not model.startswith("models/"):
        model = f"models/{model}"

    from google.api_core.client_options import ClientOptions
    from google.ai.generativelanguage_v1beta.services.generative_service.client import (
        GenerativeServiceClient,
    )
    from google.ai.generativelanguage_v1beta.types import content, generative_service

    client = GenerativeServiceClient(client_options=ClientOptions(api_key=api_key))
    request = generative_service.GenerateContentRequest(
        model=model,
        contents=[
            content.Content(
                parts=[content.Part(text=_SYSTEM_PROMPT + "\n\n" + prompt)]
            )
        ],
        generation_config=generative_service.GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )

    response = client.generate_content(request=request)
    candidates = response.candidates
    if not candidates:
        raise RuntimeError("Gemini API returned no candidates.")

    candidate_text = []
    for candidate in candidates:
        for part in candidate.content.parts:
            if getattr(part, "text", None) is not None:
                candidate_text.append(part.text)

    return "".join(candidate_text).strip()


# ============================================================================
# CLIENT-LEVEL SCORING
# ============================================================================

def parse_vector_from_response(
    response_text: str,
    expected_length: int,
    valid_scores: List[int] | tuple[int, ...] = (0, 1, 2, 3),
) -> List[int]:
    """
    Parse the model's response into one full trajectory vector.

    This function checks that:
    - the response is a JSON list
    - every item is an allowed score
    - the list length matches the number of note-to-note transitions

    Example valid response:
    [2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    """
    try:
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(text)

        # Accept both {"scores": [...]} and a plain list
        if isinstance(data, dict):
            data = data.get("scores", data.get("Scores", []))

        if not isinstance(data, list):
            raise ValueError("Model did not return a list")

        cleaned = []
        for value in data:
            score = int(value)
            if score not in valid_scores:
                raise ValueError(f"Invalid score {score}")
            cleaned.append(score)

        if len(cleaned) != expected_length:
            raise ValueError(
                f"Expected vector length {expected_length}, got {len(cleaned)}"
            )
        return cleaned
    except Exception:
        return []


def get_validated_vector_from_llm(
    prompt: str,
    expected_length: int,
    config: BaseQ1Config,
    client_id: str,
) -> List[int]:
    """
    Call the LLM, validate the returned vector, and retry once if needed.

    If the first response is empty or malformed, this function runs the same
    prompt one more time. If the second response is still invalid, it raises an
    error so the whole program stops instead of continuing with bad outputs.
    """
    if expected_length == 0:
        return []

    for attempt in (1, 2):
        raw_response = call_llm(prompt)
        estimated_vector = parse_vector_from_response(
            raw_response,
            expected_length=expected_length,
            valid_scores=config.valid_scores,
        )
        if estimated_vector:
            return estimated_vector

        if attempt == 1:
            print(
                f"Invalid LLM response for client {client_id}. "
                "Retrying once with the same prompt..."
            )

    raise RuntimeError(
        f"LLM returned an invalid trajectory vector twice for client {client_id}. "
        "Stopping program."
    )


def score_client_record(
    client_record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """
    Score one client's full note sequence.

    What this function does:
    - pulls all notes for one client
    - turns those notes into a JSON string for the prompt
    - calls the LLM once for the whole sequence
    - parses the returned vector of progress scores
    - returns one output record with the estimated vector

    If the input record already has a true scored vector, it is copied into the
    output too so the evaluation step can compare true vs estimated values.
    """
    all_notes = client_record[config.notes_key]
    client_id = str(client_record[config.client_id_key])
    notes_json_str = json.dumps(all_notes, ensure_ascii=False, indent=2)
    expected_length = max(len(all_notes) - 1, 0)

    prompt = build_prompt(notes_json_str)
    estimated_vector = get_validated_vector_from_llm(
        prompt=prompt,
        expected_length=expected_length,
        config=config,
        client_id=client_id,
    )

    scored_record = {
        config.client_id_key: client_record[config.client_id_key],
        config.notes_key: client_record[config.notes_key],
        config.pred_vector_key: estimated_vector,
    }
    if config.true_vector_key in client_record:
        scored_record[config.true_vector_key] = client_record[config.true_vector_key]
    return scored_record


def score_dataset(
    data: List[Dict[str, Any]],
    config: BaseQ1Config,
    progress_desc: str,
) -> List[Dict[str, Any]]:
    """Score every client record in a dataset and return the scored records."""
    scored = []

    for client_record in tqdm(data, desc=progress_desc):
        scored_record = score_client_record(client_record, config)
        scored.append(scored_record)

    return scored


# ============================================================================
# EVALUATION SECTION
# ============================================================================

# ============================================================================
# TODO 3 of 3 — PERFORMANCE METRICS
# ============================================================================

def compute_metrics(step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute performance metrics from the step-level comparisons.

    Metrics chosen and justification:
    ----------------------------------
    1. Quadratic Weighted Kappa (QWK) — PRIMARY metric.
       Scores are ordinal (0 < 1 < 2 < 3), so agreement metrics that account
       for ordering are appropriate. QWK penalizes disagreements proportionally
       to the square of the distance (e.g., predicting 3 when true is 0 costs
       more than predicting 1), which matches the clinical reality that large
       scoring errors are worse than small ones. Standard for ordinal rating tasks.

    2. Exact accuracy — simple sanity check on exact-match rate.

    3. Within-1 accuracy — fraction where |pred - true| <= 1.
       Given the class imbalance (0 is ~52% of labels), a model that predicts
       all 0s gets ~52% exact accuracy. Within-1 reveals whether near-misses
       are also correct.

    4. MAE — mean absolute error, provides a direct magnitude of scoring error.

    Parameters
    ----------
    step_rows : List[Dict[str, Any]]
        One dict per scored note transition. Each dict has:
            "true_score"      – Patel's hand-assigned score (int, 0–3)
            "estimated_score" – LLM's predicted score       (int, 0–3)

    Returns
    -------
    Dict[str, Any]
        Metric name → value, printed by print_evaluation().
    """
    from sklearn.metrics import cohen_kappa_score

    true_scores = [row["true_score"] for row in step_rows]
    pred_scores = [row["estimated_score"] for row in step_rows]

    n = len(true_scores)
    exact = sum(1 for t, p in zip(true_scores, pred_scores) if t == p)
    within1 = sum(1 for t, p in zip(true_scores, pred_scores) if abs(t - p) <= 1)
    mae = sum(abs(t - p) for t, p in zip(true_scores, pred_scores)) / n

    # QWK requires at least two distinct values; guard against degenerate cases
    qwk = cohen_kappa_score(true_scores, pred_scores, weights="quadratic")

    # Score distribution breakdown (useful for diagnosing bias)
    pred_dist = {s: pred_scores.count(s) for s in (0, 1, 2, 3)}

    return {
        "qwk": round(qwk, 4),
        "exact_accuracy": round(exact / n, 4),
        "within1_accuracy": round(within1 / n, 4),
        "mae": round(mae, 4),
        "n": n,
        "pred_score_distribution": pred_dist,
    }


def evaluate_predictions(
    config: Q1ALabeledConfig,
) -> Dict[str, Any]:
    """
    Compare each client's true scored_vector with the predicted
    estimated_trajectory_vector, then compute metrics and the confusion matrix.
    """
    scored_test_data = load_json(config.evaluated_output_path)
    comparisons = build_evaluation_comparisons(scored_test_data, config)
    step_rows = comparisons["step_level_comparisons"]

    metrics = compute_metrics(step_rows)
    confusion_matrix = build_confusion_matrix(step_rows, config.valid_scores)

    return {
        **metrics,
        "confusion_matrix": confusion_matrix,
    }


def print_evaluation(results: Dict[str, Any]) -> None:
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        if key == "confusion_matrix" and isinstance(value, dict):
            print("confusion_matrix:")
            print(value.get("table", ""))
        else:
            print(f"{key}: {value}")


# ============================================================================
# PIPELINES
# ============================================================================

def run_test_pipeline(config: Q1ALabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on labeled test data."""
    test_data = load_json(config.test_path)

    scored_test_data = score_dataset(
        data=test_data,
        config=config,
        progress_desc="Scoring labeled clients",
    )
    save_json(scored_test_data, config.evaluated_output_path)

    results = evaluate_predictions(config)
    print_evaluation(results)

    return scored_test_data


def run_unlabeled_pipeline(config: Q1BUnlabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on unlabeled note data and save scored outputs."""
    unlabeled_data = load_json(config.unlabeled_path)

    scored_unlabeled_data = score_dataset(
        data=unlabeled_data,
        config=config,
        progress_desc="Scoring unlabeled clients",
    )
    save_json(scored_unlabeled_data, config.output_path)

    # Q1b: write scored_notes.csv for Q2
    csv_path = ensure_parent_dir("output/q1/scored_notes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "session", "score"])
        for record in scored_unlabeled_data:
            client_id = record[config.client_id_key]
            scores = record[config.pred_vector_key]
            notes = record[config.notes_key]
            for i, score in enumerate(scores):
                # session = the starting note number of the transition
                session = notes[i][config.note_number_key]
                writer.writerow([client_id, session, score])
    print(f"Saved: {csv_path}")

    return scored_unlabeled_data


# ============================================================================
# ENTRY POINT
# ============================================================================
#
# HOW TO WORK THROUGH THIS FILE
# ──────────────────────────────
# Step 1 — implement build_prompt(), call_llm(), and compute_metrics()  [done]
# Step 2 — run run_test_pipeline(LABELED_CONFIG) to score the labeled set
#           and see your metrics + confusion matrix printed to the terminal
# Step 3 — iterate on your prompt; re-run Step 2 to compare versions
# Step 4 — once satisfied, run run_unlabeled_pipeline(UNLABELED_CONFIG)
#           to score all unlabeled clients → produces scored_notes.json + .csv for Q2
#
# TIP: test on a single client first:
#
#   import json
#   sample = load_json("data/labeled_notes.json")[0]
#   notes_str = json.dumps(sample["notes"], indent=2)
#   print(build_prompt(notes_str))           # inspect the prompt visually
#   print(call_llm(build_prompt(notes_str))) # check the raw model response
# ============================================================================

if __name__ == "__main__":
    LABELED_CONFIG = Q1ALabeledConfig(
        test_path="data/labeled_notes.json",
        evaluated_output_path="output/q1/evaluated_labeled_results.json",
    )
    UNLABELED_CONFIG = Q1BUnlabeledConfig(
        unlabeled_path="data/unlabeled_notes.json",
        output_path="output/q1/scored_notes.json",
    )

    # Step 2 — run run_test_pipeline(LABELED_CONFIG) to score the labeled set
    #           and see your metrics + confusion matrix printed to the terminal
    run_test_pipeline(LABELED_CONFIG)
