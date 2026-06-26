import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.getxapi.com"
REPLIES_ENDPOINT = "/twitter/tweet/replies"
DEFAULT_TIMEOUT = 30


class GetXAPIResponseError(ValueError):
    """Raised when GetXAPI returns an unexpected response shape."""


class GetXAPIRequestError(requests.HTTPError):
    """Raised when GetXAPI returns an HTTP error with a useful message."""


def get_headers() -> dict:
    api_key = os.getenv("GETXAPI_KEY", "").strip()

    if not api_key:
        raise ValueError("GETXAPI_KEY not found. Add it to your .env file.")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def fetch_replies(tweet_id: str, cursor: str | None = None) -> dict:
    """
    Fetch replies under a specific parent tweet.

    Returns the raw GetXAPI response, including:
    - tweetId
    - reply_count
    - has_more
    - next_cursor
    - replies
    """
    url = f"{BASE_URL}{REPLIES_ENDPOINT}"

    params = {"id": tweet_id}

    if cursor is not None:
        params["cursor"] = cursor

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )

    _raise_for_status(response)
    return response.json()


def fetch_all_replies(tweet_id: str, max_pages: int | None = None) -> list[dict]:
    """
    Fetch every available replies page for a tweet.

    Set max_pages during exploration to cap API usage.
    """
    replies = []
    cursor = None
    pages_fetched = 0

    while True:
        response = fetch_replies(tweet_id=tweet_id, cursor=cursor)
        replies.extend(_extract_replies(response))
        pages_fetched += 1

        if max_pages is not None and pages_fetched >= max_pages:
            break

        cursor = response.get("next_cursor") or response.get("nextCursor")
        has_more = response.get("has_more", response.get("hasMore", bool(cursor)))

        if not has_more or not cursor:
            break

    return replies


def _first_present(raw: dict, keys: tuple[str, ...]):
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return None


def _raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("message")

        if detail:
            message = f"{exc} - GetXAPI error: {detail}"
            raise GetXAPIRequestError(message, response=response) from exc

        raise


def _extract_replies(response: dict) -> list[dict]:
    replies = response.get("replies")

    if replies is None and isinstance(response.get("data"), dict):
        replies = response["data"].get("replies")

    if replies is None:
        return []

    if not isinstance(replies, list):
        raise GetXAPIResponseError("Expected GetXAPI 'replies' field to be a list.")

    return replies


def _extract_parent_tweet_id(response: dict) -> str | None:
    parent_tweet_id = _first_present(response, ("tweetId", "tweet_id", "id"))

    if parent_tweet_id is None and isinstance(response.get("data"), dict):
        parent_tweet_id = _first_present(response["data"], ("tweetId", "tweet_id", "id"))

    return parent_tweet_id


def normalize_reply(
    raw: dict,
    parent_tweet_id: str,
    match: str,
    event: str,
    source_account: str | None = None,
    team: str | None = None,
    player: str | None = None,
) -> dict:
    """
    Convert one raw GetXAPI reply into the project schema.
    """

    author = raw.get("author", {})

    return {
        "tweet_id": _first_present(raw, ("id", "tweetId", "tweet_id")),
        "parent_tweet_id": parent_tweet_id,
        "url": raw.get("url"),
        "timestamp": _first_present(raw, ("createdAt", "created_at", "timestamp", "date")),
        "text": raw.get("text"),

        "author_username": _first_present(author, ("userName", "username", "screen_name")),
        "author_name": author.get("name"),

        "like_count": _first_present(raw, ("likeCount", "like_count", "favorite_count")),
        "reply_count": _first_present(raw, ("replyCount", "reply_count")),
        "view_count": _first_present(raw, ("viewCount", "view_count")),

        "match": match,
        "team": team,
        "player": player,
        "event": event,
        "source_account": source_account,
        "source": "getxapi",
    }


def normalize_replies(
    response: dict,
    match: str,
    event: str,
    source_account: str | None = None,
    team: str | None = None,
    player: str | None = None,
) -> list[dict]:
    """
    Convert a full GetXAPI replies response into normalized project rows.
    """

    parent_tweet_id = _extract_parent_tweet_id(response)
    replies = _extract_replies(response)

    return [
        normalize_reply(
            raw=reply,
            parent_tweet_id=parent_tweet_id,
            match=match,
            event=event,
            source_account=source_account,
            team=team,
            player=player,
        )
        for reply in replies
    ]
