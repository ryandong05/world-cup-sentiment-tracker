import html
import re
from collections.abc import Iterable

import pandas as pd

from .entities import infer_entities, load_entity_reference

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

MEDIA_ONLY_TERMS = {"gif", "image", "photo", "pic", "video", "media"}
SPAM_PHRASES = (
    "ask grok",
    "follow me",
    "check my profile",
    "check profile",
    "dm me",
    "subscribe",
    "giveaway",
    "crypto",
    "airdrop",
    "buy now",
    "limited time",
    "order now",
    "switching to t-mobile",
    "t-mobile",
    "iphone 17",
    "free shipping",
    "delivery comes",
    "master antioxidant",
    "luxurious softness",
)
CONTEXTLESS_VALUES = {"", "none", "nan", "null", "multiple", "unknown", "n/a"}
CONTEXT_DEPENDENT_TERMS = {
    "achieve",
    "achieved",
    "always",
    "another record",
    "back",
    "breaking record",
    "breaking records",
    "he",
    "him",
    "his",
    "man",
    "no one",
    "record",
    "records",
    "this man",
}
FOOTBALL_TERMS = {
    "assist",
    "back four",
    "back line",
    "back three",
    "bench",
    "build up",
    "buildup",
    "coach",
    "comeback",
    "corner",
    "counter attack",
    "counterattack",
    "cup",
    "defensive line",
    "double pivot",
    "equalizer",
    "final",
    "finish",
    "formation",
    "football",
    "foul",
    "free kick",
    "game",
    "game plan",
    "goal",
    "head coach",
    "high line",
    "keeper",
    "lineup",
    "low block",
    "manager",
    "match",
    "midfield",
    "number 10",
    "offside",
    "overload",
    "penalty",
    "pivot",
    "player",
    "possession",
    "press",
    "pressing",
    "red",
    "red card",
    "ref",
    "referee",
    "rotation",
    "save",
    "score",
    "scored",
    "selection",
    "set piece",
    "shape",
    "soccer",
    "starting 11",
    "starting xi",
    "striker",
    "sub",
    "substitution",
    "system",
    "tactic",
    "tactical",
    "tactics",
    "team",
    "transition",
    "var",
    "winger",
    "win",
    "winner",
    "world cup",
    "yellow",
    "yellow card",
}


def clean_text(text: object) -> str:
    """Clean social-media text without applying match-specific knowledge."""
    if not isinstance(text, str):
        return ""

    cleaned = html.unescape(text)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("#", "")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)

    return cleaned.strip()


def is_usable_text(text: object, min_chars: int = 5, min_words: int = 2) -> bool:
    """Return whether text contains enough language for sentiment analysis."""
    cleaned = clean_text(text)
    words = WORD_RE.findall(cleaned)

    if len(cleaned) < min_chars:
        return False

    if len(words) < min_words:
        return False

    if cleaned.lower() in MEDIA_ONLY_TERMS:
        return False

    return True


def is_spam_text(text: object) -> bool:
    """Return whether text looks like ad, bot, or platform-promo noise."""
    cleaned = clean_text(text).lower()

    if not cleaned:
        return False

    return any(phrase in cleaned for phrase in SPAM_PHRASES)


def relevance_score(
    text: object,
    match: object = None,
    team: object = None,
    player: object = None,
    event: object = None,
    extra_terms: Iterable[str] | None = None,
    entity_confidence: int = 0,
    generic_terms: Iterable[str] | None = None,
    parent_context_confidence: int = 0,
) -> int:
    """Score whether text appears related to the known match context."""
    cleaned = clean_text(text).lower()

    if not cleaned:
        return -2

    score = 0
    spam_text = is_spam_text(cleaned)

    if not is_usable_text(cleaned) and entity_confidence <= 0:
        score -= 2

    score += entity_confidence * 2

    if spam_text:
        score -= 3

    context_terms = _context_terms(match=match, team=team, player=player, event=event)
    if extra_terms:
        context_terms.update(_normalize_term(term) for term in extra_terms)
        context_terms.discard("")

    for term in context_terms:
        if _contains_term(cleaned, term):
            score += 2

    for term in generic_terms or FOOTBALL_TERMS:
        if _contains_term(cleaned, term):
            score += 1

    if parent_context_confidence > 0 and is_usable_text(cleaned) and not spam_text:
        score += 1

        for term in CONTEXT_DEPENDENT_TERMS:
            if _contains_term(cleaned, term):
                score += 1
                break

    return score


def is_match_relevant(
    text: object,
    match: object = None,
    team: object = None,
    player: object = None,
    event: object = None,
    min_score: int = 1,
    extra_terms: Iterable[str] | None = None,
    entity_confidence: int = 0,
    generic_terms: Iterable[str] | None = None,
    parent_context_confidence: int = 0,
) -> bool:
    """Return whether text is relevant enough for match sentiment analysis."""
    return (
        relevance_score(
            text=text,
            match=match,
            team=team,
            player=player,
            event=event,
            extra_terms=extra_terms,
            entity_confidence=entity_confidence,
            generic_terms=generic_terms,
            parent_context_confidence=parent_context_confidence,
        )
        >= min_score
    )


def filter_reason(row: pd.Series, min_relevance_score: int = 1) -> str:
    """Explain why a reply is kept or filtered."""
    if row.get("is_spam_text", False):
        return "spam"

    if not row["is_usable_text"]:
        if row.get("entity_confidence", 0) <= 0:
            return "unusable_text"

    if row["relevance_score"] < min_relevance_score:
        return "low_relevance"

    return "keep"


def preprocess_replies(
    df: pd.DataFrame,
    min_relevance_score: int = 1,
    keep_audit_columns: bool = True,
    extra_terms: Iterable[str] | None = None,
    entity_reference: dict | None = None,
) -> pd.DataFrame:
    """
    Clean, score, and filter normalized reply rows for sentiment analysis.

    The input dataframe is not mutated.
    """
    processed = annotate_replies(
        df,
        min_relevance_score=min_relevance_score,
        extra_terms=extra_terms,
        entity_reference=entity_reference,
    )

    analysis_df = processed[processed["filter_reason"] == "keep"].copy()

    if not keep_audit_columns:
        analysis_df = analysis_df.drop(
            columns=["is_usable_text", "relevance_score", "filter_reason"],
        )

    return analysis_df.reset_index(drop=True)


def annotate_replies(
    df: pd.DataFrame,
    min_relevance_score: int = 1,
    extra_terms: Iterable[str] | None = None,
    entity_reference: dict | None = None,
) -> pd.DataFrame:
    """Add cleaning and filtering audit columns without dropping rows."""
    if "text" not in df.columns:
        raise ValueError("annotate_replies requires a 'text' column.")

    processed = df.copy()
    reference = entity_reference or load_entity_reference()
    generic_terms = reference.get("generic_football_terms", FOOTBALL_TERMS)

    processed["clean_text"] = processed["text"].apply(clean_text)
    processed["is_usable_text"] = processed["text"].apply(is_usable_text)
    processed["is_spam_text"] = processed["text"].apply(is_spam_text)
    processed["parent_context_confidence"] = processed.apply(
        _parent_context_confidence,
        axis=1,
    )
    entity_rows = processed.apply(
        lambda row: infer_entities(
            text=row["clean_text"],
            reference=reference,
            context_teams=_row_context_teams(row),
        ),
        axis=1,
    )
    entity_df = pd.DataFrame(entity_rows.tolist(), index=processed.index)
    processed = processed.join(entity_df)
    processed["relevance_score"] = processed.apply(
        lambda row: relevance_score(
            text=row["clean_text"],
            match=row.get("match"),
            team=row.get("team"),
            player=row.get("player"),
            event=row.get("event"),
            extra_terms=extra_terms,
            entity_confidence=row.get("entity_confidence", 0),
            generic_terms=generic_terms,
            parent_context_confidence=row.get("parent_context_confidence", 0),
        ),
        axis=1,
    )
    processed["contextual_terms"] = processed["clean_text"].apply(_matched_contextual_terms)
    processed["context_relevance_boost"] = processed.apply(
        lambda row: _context_relevance_boost(
            text=row["clean_text"],
            parent_context_confidence=row.get("parent_context_confidence", 0),
            is_spam=row.get("is_spam_text", False),
        ),
        axis=1,
    )
    processed["filter_reason"] = processed.apply(
        lambda row: filter_reason(row, min_relevance_score=min_relevance_score),
        axis=1,
    )
    processed["needs_context_review"] = processed.apply(
        _needs_context_review,
        axis=1,
    )

    return processed


def preprocessing_summary(
    raw_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    reason_column: str = "filter_reason",
) -> dict:
    """Summarize how many rows were kept and filtered."""
    if reason_column not in raw_df.columns:
        raw_df = annotate_replies(raw_df)

    reason_counts = raw_df[reason_column].value_counts(dropna=False).to_dict()

    return {
        "raw_rows": len(raw_df),
        "analysis_rows": len(analysis_df),
        "removed_rows": len(raw_df) - len(analysis_df),
        "filter_reasons": reason_counts,
    }


def _context_terms(
    match: object = None,
    team: object = None,
    player: object = None,
    event: object = None,
) -> set[str]:
    terms = set()

    for value in (team, player, event):
        term = _normalize_term(value)
        if term:
            terms.add(term)

    match_term = _normalize_term(match)
    if match_term:
        terms.add(match_term)
        for part in re.split(r"\s+(?:vs|v)\.?\s+|\s+-\s+", match_term):
            part = _normalize_term(part)
            if part:
                terms.add(part)

    return terms


def _row_context_teams(row: pd.Series) -> set[str]:
    teams = set()

    team = _normalize_term(row.get("team"))
    if team:
        teams.add(team)

    match = _normalize_term(row.get("match"))
    if match:
        for part in re.split(r"\s+(?:vs|v)\.?\s+|\s+-\s+", match):
            part = _normalize_term(part)
            if part and part != "multiple":
                teams.add(part)

    return teams


def _parent_context_confidence(row: pd.Series) -> int:
    score = 0

    if _has_context_value(row.get("match")):
        score += 1

    if _has_context_value(row.get("team")):
        score += 1

    if _has_context_value(row.get("player")):
        score += 2

    if _has_context_value(row.get("event")):
        score += 1

    return score


def _context_relevance_boost(
    text: object,
    parent_context_confidence: int,
    is_spam: bool,
) -> int:
    if parent_context_confidence <= 0 or is_spam or not is_usable_text(text):
        return 0

    boost = 1

    if _matched_contextual_terms(text):
        boost += 1

    return boost


def _matched_contextual_terms(text: object) -> list[str]:
    cleaned = clean_text(text).lower()
    return sorted(
        term
        for term in CONTEXT_DEPENDENT_TERMS
        if _contains_term(cleaned, term)
    )


def _needs_context_review(row: pd.Series) -> bool:
    if row.get("filter_reason") != "keep":
        return False

    return (
        row.get("context_relevance_boost", 0) > 0
        and row.get("entity_confidence", 0) <= 0
        and len(row.get("matched_entities", [])) == 0
    )


def _has_context_value(value: object) -> bool:
    return _normalize_term(value) not in CONTEXTLESS_VALUES


def _normalize_term(value: object) -> str:
    if not isinstance(value, str):
        return ""

    return WHITESPACE_RE.sub(" ", value.lower()).strip()


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False

    if re.fullmatch(r"[a-z0-9 ]+", term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    return term in text
