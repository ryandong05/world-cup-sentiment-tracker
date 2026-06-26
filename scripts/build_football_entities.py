import argparse
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import requests
import certifi
from bs4 import BeautifulSoup

SOURCE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
DEFAULT_OUTPUT = Path("data/reference/football_entities.json")

GENERIC_FOOTBALL_TERMS = [
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
    "goat",
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
    "yellow card",
]

TEAM_ALIAS_OVERRIDES = {
    "Algeria": ["alg", "algerian"],
    "Argentina": ["arg", "argentine", "argentinian"],
    "Australia": ["aus", "aussie", "socceroos"],
    "Austria": ["aut", "austrian"],
    "Belgium": ["bel", "belgian", "red devils"],
    "Bosnia and Herzegovina": ["bih", "bosnia", "bosnian"],
    "Brazil": ["bra", "brasil", "brazilian", "selecao", "seleção"],
    "Canada": ["can", "canadian"],
    "Cape Verde": ["cpv", "cabo verde", "cape verdean"],
    "Colombia": ["col", "colombian"],
    "Croatia": ["cro", "croatian"],
    "Curaçao": ["curacao", "cuw"],
    "Czech Republic": ["cze", "czech", "czechia"],
    "DR Congo": ["cod", "congo", "drc", "democratic republic of congo"],
    "Ecuador": ["ecu", "ecuadorian"],
    "Egypt": ["egy", "egyptian"],
    "England": ["eng", "english", "three lions"],
    "France": ["fra", "french", "les bleus"],
    "Germany": ["deu", "ger", "german"],
    "Ghana": ["gha", "ghanaian"],
    "Haiti": ["hai", "haitian"],
    "Iran": ["irn", "iranian"],
    "Iraq": ["irq", "iraqi"],
    "Ivory Coast": ["civ", "cote d'ivoire", "côte d'ivoire", "ivorian"],
    "Japan": ["jpn", "japanese"],
    "Jordan": ["jor", "jordanian"],
    "Mexico": ["mex", "mexican", "el tri"],
    "Morocco": ["mar", "moroccan"],
    "Netherlands": ["ned", "dutch", "holland", "oranje"],
    "New Zealand": ["nzl", "kiwis"],
    "Norway": ["nor", "norwegian"],
    "Panama": ["pan", "panamanian"],
    "Paraguay": ["par", "paraguayan"],
    "Portugal": ["por", "portuguese"],
    "Qatar": ["qat", "qatari"],
    "Saudi Arabia": ["ksa", "sau", "saudi", "saudis"],
    "Scotland": ["sco", "scottish"],
    "Senegal": ["sen", "senegalese"],
    "South Africa": ["rsa", "south african", "bafana bafana"],
    "South Korea": ["kor", "korea", "korean", "republic of korea"],
    "Spain": ["esp", "spanish", "la roja"],
    "Sweden": ["swe", "swedish"],
    "Switzerland": ["sui", "swiss"],
    "Tunisia": ["tun", "tunisian"],
    "Turkey": ["tur", "turkish", "turkiye", "türkiye"],
    "United States": ["usa", "usmnt", "united states of america"],
    "Uruguay": ["uru", "uruguayan"],
    "Uzbekistan": ["uzb", "uzbek"],
}

PLAYER_ALIAS_OVERRIDES = {
    "Cristiano Ronaldo": ["cr7", "cristiano", "ronaldo", "siuuu"],
    "Kylian Mbappé": ["kylian", "mbappe", "mbappé"],
    "Lionel Messi": ["leo messi", "messi"],
    "Neymar": ["neymar"],
    "Rafael Leão": ["leao", "leão", "rafael leao", "rafael leão"],
    "Vinícius Júnior": ["vini", "vini jr", "vinicius", "vinicius junior", "vinícius júnior"],
}

MANAGER_ALIAS_OVERRIDES = {
    "Carlo Ancelotti": ["ancelotti"],
    "Didier Deschamps": ["deschamps"],
    "Fabio Cannavaro": ["cannavaro"],
    "Javier Aguirre": ["aguirre"],
    "Julian Nagelsmann": ["nagelsmann"],
    "Lionel Scaloni": ["scaloni"],
    "Luis de la Fuente": ["de la fuente"],
    "Mauricio Pochettino": ["pochettino"],
    "Roberto Martínez": ["martinez", "martínez", "roberto martinez", "roberto martínez"],
    "Thomas Tuchel": ["tuchel"],
}

AMBIGUOUS_PLAYER_ALIASES = {
    "can",
    "just",
    "may",
    "will",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--input-html", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    html = args.input_html.read_text(encoding="utf-8") if args.input_html else None
    reference = build_reference(source_url=args.source_url, html=html)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    team_count = len(reference["teams"])
    player_count = sum(len(team["players"]) for team in reference["teams"].values())
    manager_count = sum(1 for team in reference["teams"].values() if team.get("manager"))
    print(
        f"Wrote {args.output} with {team_count} teams, "
        f"{manager_count} managers, and {player_count} players."
    )


def build_reference(source_url: str = SOURCE_URL, html: str | None = None) -> dict:
    if html is None:
        response = requests.get(source_url, timeout=30, verify=certifi.where())
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "html.parser")
    teams = {}

    for heading in soup.find_all("h3"):
        team_name = _clean_heading(heading.get_text(" ", strip=True))
        if team_name not in TEAM_ALIAS_OVERRIDES:
            continue

        table = _find_squad_table(heading)

        if not team_name or table is None:
            continue

        manager = _extract_manager(heading)
        players = _extract_players(table)
        if not players:
            continue

        teams[team_name] = {
            "aliases": sorted(_team_aliases(team_name)),
            "manager": (
                {
                    "name": manager,
                    "aliases": sorted(_manager_aliases(manager)),
                }
                if manager
                else None
            ),
            "players": {
                player: sorted(_player_aliases(player))
                for player in players
            },
        }

    return {
        "source": {
            "name": "2026 FIFA World Cup squads - Wikipedia",
            "url": source_url,
            "retrieved_at": datetime.now(UTC).isoformat(),
        },
        "teams": dict(sorted(teams.items())),
        "generic_football_terms": GENERIC_FOOTBALL_TERMS,
    }


def _find_squad_table(heading):
    parent = heading.parent
    node = parent.find_next_sibling()

    while node is not None:
        if node.name in {"h2", "h3"}:
            return None

        if node.name == "table" and "wikitable" in node.get("class", []):
            if _player_column_index(node) is not None:
                return node

        table = node.find("table", class_="wikitable") if hasattr(node, "find") else None
        if table is not None and _player_column_index(table) is not None:
            return table

        node = node.find_next_sibling()

    return None


def _player_column_index(table) -> int | None:
    header_row = table.find("tr")
    if header_row is None:
        return None

    headers = [
        cell.get_text(" ", strip=True).lower()
        for cell in header_row.find_all(["th", "td"])
    ]

    for index, header in enumerate(headers):
        if "player" in header:
            return index

    return None


def _extract_players(table) -> list[str]:
    player_index = _player_column_index(table)
    if player_index is None:
        return []

    players = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) <= player_index:
            continue

        player = _clean_player_name(cells[player_index].get_text(" ", strip=True))
        if player:
            players.append(player)

    return players


def _extract_manager(heading) -> str | None:
    node = heading.parent.find_next_sibling()

    while node is not None:
        if node.name in {"h2", "h3", "table"}:
            return None

        if node.name == "p" and "Coach:" in node.get_text(" ", strip=True):
            links = node.find_all("a")
            if links:
                return _clean_player_name(links[-1].get_text(" ", strip=True))

            manager = node.get_text(" ", strip=True).split("Coach:", 1)[-1]
            manager = _clean_player_name(manager)
            return manager if manager and manager.lower() != "vacant" else None

        node = node.find_next_sibling()

    return None


def _clean_heading(value: str) -> str:
    return re.sub(r"\s*\[edit\]\s*$", "", value).strip()


def _clean_player_name(value: str) -> str:
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = re.sub(r"\s*\([^)]*\)", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _team_aliases(team_name: str) -> set[str]:
    aliases = {_normalize_alias(team_name), _ascii_alias(team_name)}
    aliases.update(_normalize_alias(alias) for alias in TEAM_ALIAS_OVERRIDES.get(team_name, []))
    aliases.update(_ascii_alias(alias) for alias in TEAM_ALIAS_OVERRIDES.get(team_name, []))
    aliases.discard("")
    return aliases


def _player_aliases(player_name: str) -> set[str]:
    aliases = {_normalize_alias(player_name), _ascii_alias(player_name)}
    aliases.update(_normalize_alias(alias) for alias in PLAYER_ALIAS_OVERRIDES.get(player_name, []))
    aliases.update(_ascii_alias(alias) for alias in PLAYER_ALIAS_OVERRIDES.get(player_name, []))

    name_parts = _ascii_alias(player_name).split()
    if len(name_parts) == 1:
        aliases.add(name_parts[0])
    elif len(name_parts[-1]) >= 4 and name_parts[-1] not in AMBIGUOUS_PLAYER_ALIASES:
        aliases.add(name_parts[-1])

    aliases.discard("")
    return aliases


def _manager_aliases(manager_name: str) -> set[str]:
    aliases = {_normalize_alias(manager_name), _ascii_alias(manager_name)}
    aliases.update(_normalize_alias(alias) for alias in MANAGER_ALIAS_OVERRIDES.get(manager_name, []))
    aliases.update(_ascii_alias(alias) for alias in MANAGER_ALIAS_OVERRIDES.get(manager_name, []))

    name_parts = _ascii_alias(manager_name).split()
    if len(name_parts) > 1 and len(name_parts[-1]) >= 4:
        aliases.add(name_parts[-1])

    aliases.discard("")
    return aliases


def _normalize_alias(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _ascii_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return _normalize_alias(ascii_value)


if __name__ == "__main__":
    main()
