import html
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "football_entities.json"
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
COMMON_ALIAS_STOPWORDS = {
    "can",
    "just",
    "may",
    "will",
}
REPEATED_ALIAS_PATTERNS = {
    "siuuu": r"(?<![a-z0-9])siu+(?![a-z0-9])",
}


def load_entity_reference(path: str | Path | None = None) -> dict:
    """Load the football entity reference data."""
    reference_path = Path(path) if path else DEFAULT_REFERENCE_PATH
    return json.loads(reference_path.read_text(encoding="utf-8"))


def extract_entities(
    text: object,
    reference: dict | None = None,
    context_teams: Iterable[str] | None = None,
) -> dict:
    """Extract football entity matches from text."""
    return infer_entities(
        text=text,
        reference=reference,
        context_teams=context_teams,
    )


def infer_entities(
    text: object,
    reference: dict | None = None,
    context_teams: Iterable[str] | None = None,
) -> dict:
    """
    Infer teams, players, and managers from reply text.

    Ambiguous aliases are only linked to entities when row context resolves them.
    """
    reference = reference or load_entity_reference()
    normalized_text = normalize_text(text)
    alias_index = _build_alias_index(reference)
    context = {_normalize_alias(team) for team in context_teams or [] if team}

    matched_entities = set()
    inferred_teams = set()
    inferred_players = set()
    inferred_managers = set()
    entity_confidence = 0

    for alias in sorted(alias_index, key=len, reverse=True):
        if not _contains_alias(normalized_text, alias):
            continue

        records = alias_index[alias]
        resolved_records = _resolve_records(records, context)

        matched_entities.add(alias)
        entity_confidence += 1 if not resolved_records else len(resolved_records)

        for record in resolved_records:
            team = record.get("team")
            if team:
                inferred_teams.add(team)

            if record["type"] == "player":
                inferred_players.add(record["name"])
            elif record["type"] == "manager":
                inferred_managers.add(record["name"])

    return {
        "matched_entities": sorted(matched_entities),
        "inferred_teams": sorted(inferred_teams),
        "inferred_players": sorted(inferred_players),
        "inferred_managers": sorted(inferred_managers),
        "entity_confidence": entity_confidence,
    }


def normalize_text(text: object) -> str:
    """Normalize text and accents for entity matching."""
    if not isinstance(text, str):
        return ""

    normalized = html.unescape(text)
    normalized = HTML_TAG_RE.sub(" ", normalized)
    normalized = URL_RE.sub(" ", normalized)
    normalized = MENTION_RE.sub(" ", normalized)
    normalized = normalized.replace("#", "")
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = WHITESPACE_RE.sub(" ", normalized)

    return normalized.strip()


def _build_alias_index(reference: dict) -> dict[str, list[dict]]:
    alias_index = defaultdict(list)

    for term in reference.get("generic_football_terms", []):
        alias = _normalize_alias(term)
        if alias and alias not in COMMON_ALIAS_STOPWORDS:
            alias_index[alias].append({"type": "generic", "name": term, "team": None})

    for team_name, team_data in reference.get("teams", {}).items():
        for alias in team_data.get("aliases", []):
            alias = _normalize_alias(alias)
            if alias and alias not in COMMON_ALIAS_STOPWORDS:
                alias_index[alias].append({"type": "team", "name": team_name, "team": team_name})

        manager = team_data.get("manager") or {}
        manager_name = manager.get("name")
        for alias in manager.get("aliases", []):
            alias = _normalize_alias(alias)
            if alias and manager_name and alias not in COMMON_ALIAS_STOPWORDS:
                alias_index[alias].append(
                    {"type": "manager", "name": manager_name, "team": team_name}
                )

        for player_name, aliases in team_data.get("players", {}).items():
            for alias in aliases:
                alias = _normalize_alias(alias)
                if alias and alias not in COMMON_ALIAS_STOPWORDS:
                    alias_index[alias].append(
                        {"type": "player", "name": player_name, "team": team_name}
                    )

    return {
        alias: _dedupe_records(records)
        for alias, records in alias_index.items()
    }


def _dedupe_records(records: list[dict]) -> list[dict]:
    unique_records = []
    seen = set()

    for record in records:
        key = (record["type"], record.get("name"), record.get("team"))
        if key in seen:
            continue

        seen.add(key)
        unique_records.append(record)

    return unique_records


def _resolve_records(records: list[dict], context_teams: set[str]) -> list[dict]:
    entity_records = [record for record in records if record["type"] != "generic"]

    if not entity_records:
        return []

    if len(entity_records) == 1:
        return entity_records

    if context_teams:
        contextual = [
            record
            for record in entity_records
            if _normalize_alias(record.get("team")) in context_teams
        ]
        if contextual:
            return contextual

    return []


def _normalize_alias(value: object) -> str:
    if not isinstance(value, str):
        return ""

    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()

    return WHITESPACE_RE.sub(" ", normalized).strip()


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False

    if alias in REPEATED_ALIAS_PATTERNS:
        return re.search(REPEATED_ALIAS_PATTERNS[alias], text) is not None

    pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
    return re.search(pattern, text) is not None
