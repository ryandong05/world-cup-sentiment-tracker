import argparse
import csv
import json
import re
import unicodedata
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import certifi
import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
DEFAULT_OUTPUT = Path("data/reference/matches_sample.csv")
DEFAULT_ENTITY_REFERENCE = Path("data/reference/football_entities.json")

STAGE_MAP = {
    "Round of 32": "Round of 32",
    "Round of 16": "Round of 16",
    "Quarterfinals": "Quarterfinals",
    "Semifinals": "Semifinals",
    "Match for third place": "Third Place",
    "Final": "Final",
}
MANUAL_TEAM_ALIASES = {
    "cabo verde": "Cape Verde",
    "congo dr": "DR Congo",
    "cote d ivoire": "Ivory Coast",
    "côte d ivoire": "Ivory Coast",
    "czechia": "Czech Republic",
    "ir iran": "Iran",
    "korea republic": "South Korea",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "usa": "United States",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--input-html", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--entity-reference", type=Path, default=DEFAULT_ENTITY_REFERENCE)
    args = parser.parse_args()

    html = args.input_html.read_text(encoding="utf-8") if args.input_html else None
    rows = build_schedule(
        source_url=args.source_url,
        html=html,
        entity_reference_path=args.entity_reference,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as schedule_file:
        writer = csv.DictWriter(
            schedule_file,
            fieldnames=[
                "match_id",
                "tournament",
                "stage",
                "home_team",
                "away_team",
                "kickoff_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} with {len(rows)} matches.")


def build_schedule(
    source_url: str = SOURCE_URL,
    html: str | None = None,
    entity_reference_path: Path = DEFAULT_ENTITY_REFERENCE,
) -> list[dict]:
    if html is None:
        response = requests.get(source_url, timeout=30, verify=certifi.where())
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "html.parser")
    canonicalizer = TeamCanonicalizer(entity_reference_path)
    rows = []

    for football_box in soup.select("div.footballbox"):
        fixture = _parse_football_box(football_box, canonicalizer)
        if fixture:
            rows.append(fixture)

    rows.sort(key=lambda row: (row["kickoff_utc"], row["home_team"], row["away_team"]))
    for index, row in enumerate(rows, start=1):
        row["match_id"] = f"2026_M{index:03d}"

    return rows


def _parse_football_box(football_box, canonicalizer) -> dict | None:
    table = football_box.find("table", class_="fevent")
    metadata = football_box.find("div", class_="fleft")

    if table is None or metadata is None:
        return None

    home_team = _extract_team(table, "fhome")
    away_team = _extract_team(table, "faway")
    kickoff_utc = _extract_kickoff_utc(metadata.get_text(" ", strip=True))

    if not home_team or not away_team or kickoff_utc is None:
        return None

    return {
        "match_id": None,
        "tournament": "World Cup",
        "stage": _infer_stage(football_box),
        "home_team": canonicalizer.canonicalize(home_team),
        "away_team": canonicalizer.canonicalize(away_team),
        "kickoff_utc": kickoff_utc.isoformat().replace("+00:00", "Z"),
    }


def _extract_team(table, class_name: str) -> str:
    cell = table.find(class_=class_name)
    if cell is None:
        return ""

    itemprop_name = cell.find(attrs={"itemprop": "name"})
    raw = itemprop_name.get_text(" ", strip=True) if itemprop_name else cell.get_text(" ", strip=True)
    return _clean_text(raw)


def _extract_kickoff_utc(metadata_text: str) -> datetime | None:
    date_match = re.search(r"\(\s*(\d{4}-\d{2}-\d{2})\s*\)", metadata_text)
    time_match = re.search(
        r"(\d{1,2}:\d{2})\s*([ap])\.m\.\s*UTC([+\-−]\d{1,2})(?::(\d{2}))?",
        metadata_text,
        re.IGNORECASE,
    )

    if not date_match or not time_match:
        return None

    date_part = date_match.group(1)
    time_part = time_match.group(1)
    meridiem = time_match.group(2).lower()
    offset_hours = int(time_match.group(3).replace("−", "-"))
    offset_minutes = int(time_match.group(4) or 0)

    hour, minute = [int(part) for part in time_part.split(":")]
    if meridiem == "p" and hour != 12:
        hour += 12
    elif meridiem == "a" and hour == 12:
        hour = 0

    offset_sign = 1 if offset_hours >= 0 else -1
    local_timezone = timezone(
        timedelta(
            hours=offset_hours,
            minutes=offset_sign * offset_minutes,
        )
    )
    local_time = datetime.fromisoformat(date_part).replace(
        hour=hour,
        minute=minute,
        tzinfo=local_timezone,
    )

    return local_time.astimezone(UTC)


def _infer_stage(football_box) -> str:
    heading = football_box.find_previous(["h2", "h3"])
    heading_text = _clean_heading(heading.get_text(" ", strip=True)) if heading else ""

    if re.fullmatch(r"Group [A-L]", heading_text):
        return "Group Stage"

    return STAGE_MAP.get(heading_text, heading_text or "Unknown")


def _clean_heading(value: str) -> str:
    return re.sub(r"\s*\[edit\]\s*$", "", value).strip()


def _clean_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


class TeamCanonicalizer:
    def __init__(self, entity_reference_path: Path):
        self.aliases = dict(MANUAL_TEAM_ALIASES)

        if entity_reference_path.exists():
            reference = json.loads(entity_reference_path.read_text(encoding="utf-8"))
            for team_name, team_data in reference.get("teams", {}).items():
                self.aliases[_normalize_alias(team_name)] = team_name
                for alias in team_data.get("aliases", []):
                    self.aliases[_normalize_alias(alias)] = team_name

    def canonicalize(self, value: str) -> str:
        normalized = _normalize_alias(value)
        return self.aliases.get(normalized, value)


def _normalize_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


if __name__ == "__main__":
    main()
