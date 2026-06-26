import ast
import math
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_SCHEDULE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "matches_sample.csv"
)
REQUIRED_SCHEDULE_COLUMNS = {
    "match_id",
    "tournament",
    "stage",
    "home_team",
    "away_team",
    "kickoff_utc",
}
CONTEXTLESS_VALUES = {"", "none", "nan", "null", "multiple", "unknown", "n/a"}


def load_match_schedule(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate a match schedule reference table."""
    schedule_path = Path(path) if path else DEFAULT_SCHEDULE_PATH
    schedule_df = pd.read_csv(schedule_path)

    missing = REQUIRED_SCHEDULE_COLUMNS - set(schedule_df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Match schedule missing required columns: {missing_columns}")

    schedule_df = schedule_df.copy()
    schedule_df["kickoff_utc"] = pd.to_datetime(
        schedule_df["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    if schedule_df["kickoff_utc"].isna().any():
        raise ValueError("Match schedule contains invalid kickoff_utc values.")

    return schedule_df


def find_candidate_matches(
    teams: Iterable[str],
    timestamp,
    schedule_df: pd.DataFrame,
    window_hours: int | float | None = 48,
) -> pd.DataFrame:
    """Find scheduled matches involving any inferred team near a timestamp."""
    normalized_teams = {_normalize_team(team) for team in teams if _normalize_team(team)}
    schedule = schedule_df.copy()

    if not normalized_teams:
        return schedule.iloc[0:0].copy()

    schedule["_home_team_norm"] = schedule["home_team"].apply(_normalize_team)
    schedule["_away_team_norm"] = schedule["away_team"].apply(_normalize_team)
    schedule["matched_teams"] = schedule.apply(
        lambda row: sorted(
            {
                row["home_team"]
                for team in normalized_teams
                if team == row["_home_team_norm"]
            }
            | {
                row["away_team"]
                for team in normalized_teams
                if team == row["_away_team_norm"]
            }
        ),
        axis=1,
    )
    schedule["matched_team_count"] = schedule["matched_teams"].apply(len)
    candidates = schedule[schedule["matched_team_count"] > 0].copy()

    timestamp = parse_timestamp(timestamp)
    if timestamp is not None:
        candidates["time_delta_hours"] = (
            candidates["kickoff_utc"].sub(timestamp).abs().dt.total_seconds() / 3600
        )
        if window_hours is not None:
            candidates = candidates[candidates["time_delta_hours"] <= window_hours].copy()
    else:
        candidates["time_delta_hours"] = math.nan

    return candidates.drop(columns=["_home_team_norm", "_away_team_norm"]).reset_index(drop=True)


def score_match_candidate(row, match_row, timestamp) -> float:
    """Score one schedule candidate for a reply row."""
    matched_team_count = int(match_row.get("matched_team_count", 0))
    score = matched_team_count * 100

    if matched_team_count >= 2:
        score += 50

    timestamp = parse_timestamp(timestamp)
    if timestamp is not None:
        delta_hours = abs((match_row["kickoff_utc"] - timestamp).total_seconds()) / 3600
        score += max(0, 48 - delta_hours)

    if _has_parent_match_overlap(row, match_row):
        score += 25

    return score


def link_reply_to_match(
    row,
    schedule_df: pd.DataFrame,
    window_hours: int | float | None = 48,
    clear_gap_hours: int | float = 6,
) -> dict:
    """Link one preprocessed reply row to the most likely scheduled match."""
    timestamp, timestamp_source = _best_timestamp(row)
    teams = _row_candidate_teams(row)

    empty_result = {
        "linked_match_id": None,
        "linked_match": None,
        "linked_match_confidence": "none",
        "linked_match_method": "no_team_context",
        "match_candidates": [],
    }

    if not teams:
        return empty_result

    candidates = find_candidate_matches(
        teams=teams,
        timestamp=timestamp,
        schedule_df=schedule_df,
        window_hours=window_hours,
    )

    if candidates.empty:
        empty_result["linked_match_method"] = "no_nearby_match"
        return empty_result

    candidates = candidates.copy()
    candidates["match_score"] = candidates.apply(
        lambda candidate: score_match_candidate(row, candidate, timestamp),
        axis=1,
    )
    candidates = candidates.sort_values(
        by=["match_score", "matched_team_count", "time_delta_hours"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_payload = _candidate_payload(candidates)

    top = candidates.iloc[0]
    second = candidates.iloc[1] if len(candidates) > 1 else None
    method = _link_method(top, timestamp_source)

    if int(top["matched_team_count"]) >= 2:
        return _linked_result(top, "high", method, candidate_payload)

    if len(candidates) == 1:
        confidence = "high" if timestamp is not None else "low"
        return _linked_result(top, confidence, method, candidate_payload)

    if timestamp is None:
        return _ambiguous_result(candidate_payload, "ambiguous_without_timestamp")

    top_delta = float(top["time_delta_hours"])
    second_delta = float(second["time_delta_hours"])

    if top_delta <= 24 and second_delta - top_delta >= clear_gap_hours:
        return _linked_result(top, "medium", method, candidate_payload)

    return _ambiguous_result(candidate_payload, "ambiguous_candidates")


def link_replies_to_matches(
    df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    window_hours: int | float | None = 48,
    clear_gap_hours: int | float = 6,
) -> pd.DataFrame:
    """Add match-linking columns to a dataframe of preprocessed replies."""
    linked = df.copy()
    link_rows = linked.apply(
        lambda row: link_reply_to_match(
            row=row,
            schedule_df=schedule_df,
            window_hours=window_hours,
            clear_gap_hours=clear_gap_hours,
        ),
        axis=1,
    )
    link_df = pd.DataFrame(link_rows.tolist(), index=linked.index)

    return linked.join(link_df)


def parse_timestamp(value):
    """Parse timestamps from API strings or ISO values as UTC pandas timestamps."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None

    return timestamp


def _best_timestamp(row) -> tuple[pd.Timestamp | None, str]:
    for column in (
        "parent_tweet_timestamp",
        "parent_timestamp",
        "source_timestamp",
        "tweet_timestamp",
        "timestamp",
    ):
        if column in row:
            timestamp = parse_timestamp(row.get(column))
            if timestamp is not None:
                return timestamp, column

    return None, "none"


def _row_candidate_teams(row) -> list[str]:
    teams = []

    for column in ("inferred_teams", "team"):
        if column in row:
            teams.extend(_coerce_list(row.get(column)))

    match = row.get("match") if "match" in row else None
    if _has_context_value(match):
        teams.extend(_teams_from_match_string(match))

    return sorted({_canonical_team(team) for team in teams if _canonical_team(team)})


def _coerce_list(value) -> list[str]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if _has_context_value(item)]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in CONTEXTLESS_VALUES:
            return []

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None

            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed if _has_context_value(item)]

        return [stripped]

    return [str(value)]


def _teams_from_match_string(value) -> list[str]:
    if not isinstance(value, str):
        return []

    parts = re.split(r"\s+(?:vs|v)\.?\s+|\s+-\s+", value)
    return [part.strip() for part in parts if _has_context_value(part)]


def _has_parent_match_overlap(row, match_row) -> bool:
    match_teams = {
        _normalize_team(match_row.get("home_team")),
        _normalize_team(match_row.get("away_team")),
    }
    parent_teams = {
        _normalize_team(team)
        for team in _teams_from_match_string(row.get("match"))
    }

    return bool(parent_teams & match_teams)


def _candidate_payload(candidates: pd.DataFrame) -> list[dict]:
    payload = []

    for _, candidate in candidates.iterrows():
        payload.append(
            {
                "match_id": candidate["match_id"],
                "match": _match_label(candidate),
                "matched_teams": candidate["matched_teams"],
                "kickoff_utc": candidate["kickoff_utc"].isoformat(),
                "time_delta_hours": (
                    None
                    if pd.isna(candidate["time_delta_hours"])
                    else round(float(candidate["time_delta_hours"]), 3)
                ),
                "score": round(float(candidate["match_score"]), 3),
            }
        )

    return payload


def _linked_result(match_row, confidence: str, method: str, candidates: list[dict]) -> dict:
    return {
        "linked_match_id": match_row["match_id"],
        "linked_match": _match_label(match_row),
        "linked_match_confidence": confidence,
        "linked_match_method": method,
        "match_candidates": candidates,
    }


def _ambiguous_result(candidates: list[dict], method: str) -> dict:
    return {
        "linked_match_id": None,
        "linked_match": None,
        "linked_match_confidence": "ambiguous",
        "linked_match_method": method,
        "match_candidates": candidates,
    }


def _link_method(match_row, timestamp_source: str) -> str:
    if int(match_row.get("matched_team_count", 0)) >= 2:
        return f"both_teams_and_{timestamp_source}"

    return f"team_entity_and_{timestamp_source}"


def _match_label(match_row) -> str:
    return f"{match_row['home_team']} vs {match_row['away_team']}"


def _has_context_value(value: object) -> bool:
    return _normalize_team(value) not in CONTEXTLESS_VALUES


def _canonical_team(value: object) -> str:
    if not isinstance(value, str):
        return ""

    return re.sub(r"\s+", " ", value).strip()


def _normalize_team(value: object) -> str:
    if not isinstance(value, str):
        return ""

    return re.sub(r"\s+", " ", value.lower()).strip()
