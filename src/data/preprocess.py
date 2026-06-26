import html
import re
from collections.abc import Iterable

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

MEDIA_ONLY_TERMS = {"gif", "image", "photo", "pic", "video", "media"}
SPAM_PHRASES = (
    "follow me",
    "check my profile",
    "check profile",
    "dm me",
    "subscribe",
    "giveaway",
    "crypto",
    "airdrop",
)
FOOTBALL_TERMS = {
    "assist",
    "comeback",
    "cup",
    "equalizer",
    "final",
    "finish",
    "football",
    "foul",
    "game",
    "goal",
    "keeper",
    "match",
    "penalty",
    "player",
    "red",
    "ref",
    "referee",
    "save",
    "score",
    "scored",
    "soccer",
    "team",
    "var",
    "win",
    "winner",
    "world cup",
    "yellow",
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


def relevance_score(
    text: object,
    match: object = None,
    team: object = None,
    player: object = None,
    event: object = None,
    extra_terms: Iterable[str] | None = None,
) -> int:
    """Score whether text appears related to the known match context."""
    cleaned = clean_text(text).lower()

    if not cleaned:
        return -2

    score = 0

    if not is_usable_text(cleaned):
        score -= 2

    for phrase in SPAM_PHRASES:
        if phrase in cleaned:
            score -= 2

    context_terms = _context_terms(match=match, team=team, player=player, event=event)
    if extra_terms:
        context_terms.update(_normalize_term(term) for term in extra_terms)
        context_terms.discard("")

    for term in context_terms:
        if _contains_term(cleaned, term):
            score += 2

    for term in FOOTBALL_TERMS:
        if _contains_term(cleaned, term):
            score += 1

    return score


def is_match_relevant(
    text: object,
    match: object = None,
    team: object = None,
    player: object = None,
    event: object = None,
    min_score: int = 1,
    extra_terms: Iterable[str] | None = None,
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
        )
        >= min_score
    )


def filter_reason(row: pd.Series, min_relevance_score: int = 1) -> str:
    """Explain why a reply is kept or filtered."""
    if not row["is_usable_text"]:
        return "unusable_text"

    if row["relevance_score"] < min_relevance_score:
        return "low_relevance"

    return "keep"


def preprocess_replies(
    df: pd.DataFrame,
    min_relevance_score: int = 1,
    keep_audit_columns: bool = True,
    extra_terms: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Clean, score, and filter normalized reply rows for sentiment analysis.

    The input dataframe is not mutated.
    """
    processed = annotate_replies(
        df,
        min_relevance_score=min_relevance_score,
        extra_terms=extra_terms,
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
) -> pd.DataFrame:
    """Add cleaning and filtering audit columns without dropping rows."""
    if "text" not in df.columns:
        raise ValueError("annotate_replies requires a 'text' column.")

    processed = df.copy()
    processed["clean_text"] = processed["text"].apply(clean_text)
    processed["is_usable_text"] = processed["text"].apply(is_usable_text)
    processed["relevance_score"] = processed.apply(
        lambda row: relevance_score(
            text=row["text"],
            match=row.get("match"),
            team=row.get("team"),
            player=row.get("player"),
            event=row.get("event"),
            extra_terms=extra_terms,
        ),
        axis=1,
    )
    processed["filter_reason"] = processed.apply(
        lambda row: filter_reason(row, min_relevance_score=min_relevance_score),
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
