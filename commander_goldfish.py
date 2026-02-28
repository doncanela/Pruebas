#!/usr/bin/env python3
"""
Commander Goldfish Simulator

Simulates 10,000 goldfish starts to determine on which turn you can play your
commander.  A "goldfish" start means playing without an opponent, so the
simulation focuses purely on your own mana development.

Usage:
    python commander_goldfish.py <decklist.txt>

Decklist format (plain text, one card per line):

    Commander: Atraxa, Praetors' Voice
    1 Sol Ring
    1 Arcane Signet
    36 Forest
    ...

The "Commander:" line is required.  All other lines follow the standard
"<count> <card name>" format used by Moxfield, EDHREC, and similar sites.
Lines starting with "#" are treated as comments and ignored.

Card data is fetched automatically from the Scryfall API on the first run and
cached locally in .card_cache.json for subsequent runs.
"""

import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SIMULATIONS: int = 10_000
STARTING_HAND_SIZE: int = 7
COMMANDER_TAX: int = 2      # Additional generic mana per previous commander cast
MAX_TURNS: int = 20         # Stop each simulation after this many turns
SCRYFALL_API: str = "https://api.scryfall.com/cards/named"
CACHE_FILE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".card_cache.json")


# ---------------------------------------------------------------------------
# Card data
# ---------------------------------------------------------------------------

class Card:
    """A single Magic: The Gathering card loaded with data from Scryfall."""

    def __init__(self, name: str, data: Dict) -> None:
        self.name: str = data.get("name", name)
        self.type_line: str = data.get("type_line", "")
        self.mana_cost: str = data.get("mana_cost", "")
        self.cmc: int = int(data.get("cmc", 0))
        self.oracle_text: str = data.get("oracle_text", "")
        self.produced_mana: List[str] = data.get("produced_mana", [])

        self.is_land: bool = "Land" in self.type_line
        self.is_mana_rock: bool = (
            "Artifact" in self.type_line
            and not self.is_land
            and bool(self.produced_mana)
        )
        self.is_mana_dork: bool = (
            "Creature" in self.type_line
            and not self.is_land
            and bool(self.produced_mana)
        )

    def mana_production(self) -> int:
        """
        Estimate how much mana this permanent produces when tapped, based on
        its oracle text.  Falls back to 1 if no "Add {X}..." pattern is found.
        """
        add_match = re.search(r"Add (.+?)\.", self.oracle_text, re.IGNORECASE)
        if add_match:
            symbols = re.findall(r"\{[WUBRGCwubrgc\d/]+\}", add_match.group(1))
            if symbols:
                return len(symbols)
        # Default: produce 1 mana (correct for most basic lands and signet-like rocks)
        return 1

    def __repr__(self) -> str:
        return f"Card({self.name!r}, cmc={self.cmc})"


# ---------------------------------------------------------------------------
# Scryfall card cache
# ---------------------------------------------------------------------------

def _load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(cache: Dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def _fetch_from_scryfall(name: str) -> Optional[Dict]:
    """Fetch card data from the Scryfall API (fuzzy match on name)."""
    url = f"{SCRYFALL_API}?fuzzy={urllib.parse.quote(name)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CommanderGoldfishSimulator/1.0 (educational tool)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        time.sleep(0.05)  # Stay well within Scryfall's 10 req/s rate limit

        # Double-faced cards: use front face for type/cost, card-level for cmc
        if data.get("layout") in ("transform", "modal_dfc", "reversible_card"):
            face = data.get("card_faces", [{}])[0]
            return {
                "name": data.get("name", name),
                "type_line": face.get("type_line", data.get("type_line", "")),
                "mana_cost": face.get("mana_cost", data.get("mana_cost", "")),
                "cmc": data.get("cmc", 0),
                "produced_mana": data.get("produced_mana", []),
                "oracle_text": face.get("oracle_text", data.get("oracle_text", "")),
            }

        return {
            "name": data.get("name", name),
            "type_line": data.get("type_line", ""),
            "mana_cost": data.get("mana_cost", ""),
            "cmc": data.get("cmc", 0),
            "produced_mana": data.get("produced_mana", []),
            "oracle_text": data.get("oracle_text", ""),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Could not fetch '{name}' from Scryfall: {exc}", file=sys.stderr)
        return None


def get_card(name: str, cache: Dict) -> Card:
    """
    Return a Card object for *name*, consulting the local cache first and
    falling back to the Scryfall API when needed.  Unknown cards are returned
    with empty / zero metadata so the simulation can continue.
    """
    key = name.lower()
    if key not in cache:
        print(f"  Fetching: {name} …", file=sys.stderr)
        data = _fetch_from_scryfall(name)
        if data is None:
            data = {"name": name, "type_line": "", "mana_cost": "", "cmc": 0,
                    "produced_mana": [], "oracle_text": ""}
        cache[key] = data
    return Card(name, cache[key])


# ---------------------------------------------------------------------------
# Deck list parser
# ---------------------------------------------------------------------------

def parse_decklist(filepath: str) -> Tuple[str, List[str]]:
    """
    Parse a plain-text decklist.

    Returns:
        (commander_name, flat_list_of_card_names)

    The commander must be declared on a line that starts with "Commander:".
    All other non-blank, non-comment lines must be "<count> <name>".
    """
    commander_name: Optional[str] = None
    cards: List[str] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Commander declaration
            if line.lower().startswith("commander:"):
                commander_name = line.split(":", 1)[1].strip()
                continue

            # "<count> <card name>" with optional "[Commander]" tag
            match = re.match(r"^(\d+)\s+(.+?)(\s+\[Commander\])?\s*$", line, re.IGNORECASE)
            if not match:
                print(f"Warning: Skipping unrecognised line: {line!r}", file=sys.stderr)
                continue

            count = int(match.group(1))
            card_name = match.group(2).strip()
            is_tagged_commander = bool(match.group(3))

            if is_tagged_commander and commander_name is None:
                commander_name = card_name

            cards.extend([card_name] * count)

    if not commander_name:
        raise ValueError(
            "No commander found in the decklist.\n"
            "Add a line like:  Commander: Your Commander Name"
        )

    return commander_name, cards


# ---------------------------------------------------------------------------
# Deck builder
# ---------------------------------------------------------------------------

def build_deck(
    commander_name: str, card_names: List[str], cache: Dict
) -> Tuple[Card, List[Card]]:
    """
    Resolve all card names via cache / Scryfall and return
    (commander_card, library_cards).

    The commander is excluded from the library (it lives in the command zone).
    """
    # Fetch unique cards in sorted order to make progress output deterministic
    unique_names = sorted({commander_name} | set(card_names))
    card_objects: Dict[str, Card] = {}
    for name in unique_names:
        card_objects[name] = get_card(name, cache)

    _save_cache(cache)

    commander = card_objects[commander_name]

    library: List[Card] = []
    for name in card_names:
        # Skip if this card is the commander (it stays in command zone)
        if name.lower() == commander_name.lower():
            continue
        library.append(card_objects[name])

    return commander, library


# ---------------------------------------------------------------------------
# Game simulation
# ---------------------------------------------------------------------------

class GameState:
    """Simulates a single goldfish game from opening hand to commander cast."""

    def __init__(self, commander: Card, library: List[Card]) -> None:
        self.commander = commander
        self.commander_casts: int = 0  # Times commander has been cast this game

        # Shuffle and deal opening hand (no mulligans for speed)
        shuffled = library[:]
        random.shuffle(shuffled)
        self.hand: List[Card] = shuffled[:STARTING_HAND_SIZE]
        self.library: List[Card] = shuffled[STARTING_HAND_SIZE:]

        self.lands_in_play: List[Card] = []
        self.mana_sources_in_play: List[Card] = []  # Non-land mana producers
        self.turn: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _commander_cost(self) -> int:
        """Commander's current cost including commander tax."""
        return self.commander.cmc + COMMANDER_TAX * self.commander_casts

    def _total_mana(self) -> int:
        """Total mana available from all permanents in play."""
        return (
            sum(c.mana_production() for c in self.lands_in_play)
            + sum(c.mana_production() for c in self.mana_sources_in_play)
        )

    # ------------------------------------------------------------------
    # Turn simulation
    # ------------------------------------------------------------------

    def play_turn(self) -> bool:
        """
        Simulate one turn.

        Actions performed (in order):
          1. Draw a card.
          2. Tally mana from permanents already in play.
          3. Play a land if one is in hand (adds its production to the pool).
          4. Play mana rocks / mana dorks cheapest-first, spending mana for
             each cost and adding the card's production back to the pool.
          5. Check whether the commander can be cast from the remaining pool.

        The local *mana_pool* variable accurately tracks mana available at
        every point in the turn (spent mana is subtracted, gained mana is
        added), so the commander-cast check at step 5 is always correct.

        Returns True if the commander was cast this turn.
        """
        self.turn += 1

        # 1. Draw
        if self.library:
            self.hand.append(self.library.pop(0))

        # 2. Tally mana from permanents already in play
        mana_pool: int = self._total_mana()

        # 3. Play a land
        for card in self.hand:
            if card.is_land:
                self.hand.remove(card)
                self.lands_in_play.append(card)
                mana_pool += card.mana_production()
                break

        # 4. Play mana rocks / dorks greedily (cheapest first)
        changed = True
        while changed:
            changed = False
            candidates = [
                c for c in self.hand
                if (c.is_mana_rock or c.is_mana_dork) and c.cmc <= mana_pool
            ]
            if not candidates:
                break
            candidates.sort(key=lambda c: c.cmc)
            for rock in candidates:
                if rock.cmc <= mana_pool:
                    mana_pool -= rock.cmc               # pay the cost
                    mana_pool += rock.mana_production() # gain its production
                    self.hand.remove(rock)
                    self.mana_sources_in_play.append(rock)
                    changed = True
                    break  # restart to re-sort candidates

        # 5. Can we cast the commander?
        if mana_pool >= self._commander_cost():
            self.commander_casts += 1
            return True

        return False

    def simulate(self, max_turns: int = MAX_TURNS) -> Optional[int]:
        """
        Run the game until the commander is cast or *max_turns* is reached.

        Returns the turn number on which the commander was cast, or None.
        """
        for _ in range(max_turns):
            if self.play_turn():
                return self.turn
        return None


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_games(
    commander: Card,
    deck: List[Card],
    num_simulations: int = NUM_SIMULATIONS,
) -> List[Optional[int]]:
    """Run *num_simulations* independent game simulations and return the list
    of turns on which the commander was cast (None = not cast within MAX_TURNS)."""
    results: List[Optional[int]] = []
    for i in range(num_simulations):
        if (i + 1) % 1_000 == 0:
            print(
                f"  Simulating … {i + 1:,}/{num_simulations:,}",
                end="\r",
                file=sys.stderr,
            )
        game = GameState(commander, deck)
        results.append(game.simulate())
    print(file=sys.stderr)  # newline after progress output
    return results


# ---------------------------------------------------------------------------
# Statistics & report
# ---------------------------------------------------------------------------

def print_statistics(
    commander_name: str,
    commander_cmc: int,
    results: List[Optional[int]],
) -> None:
    """Print a human-readable simulation report to stdout."""
    total = len(results)
    cast_results = [r for r in results if r is not None]
    never_cast = total - len(cast_results)

    SEP = "=" * 62

    print(f"\n{SEP}")
    print("  Commander Goldfish Simulation Results")
    print(SEP)
    print(f"  Commander : {commander_name}")
    print(f"  CMC       : {commander_cmc}")
    print(f"  Games     : {total:,}")
    print(SEP)

    if not cast_results:
        print("\n  Commander was never cast within the simulation window.\n")
        return

    turn_counts: Counter = Counter(cast_results)

    # --- Turn distribution bar chart ---
    print("\n  Turn Distribution")
    print(f"  {'-'*58}")
    max_count = max(turn_counts.values()) if turn_counts else 1
    bar_scale = 40 / max_count  # max bar width = 40 chars
    for turn in range(1, MAX_TURNS + 1):
        count = turn_counts.get(turn, 0)
        if count == 0:
            continue
        pct = count / total * 100
        bar = "█" * int(count * bar_scale)
        print(f"  Turn {turn:2d}: {count:5,}  ({pct:5.1f}%)  {bar}")
    if never_cast:
        pct = never_cast / total * 100
        print(f"  Never : {never_cast:5,}  ({pct:5.1f}%)")

    # --- Summary statistics ---
    sorted_results = sorted(cast_results)
    n = len(sorted_results)
    mean = sum(sorted_results) / n
    median = (
        (sorted_results[n // 2 - 1] + sorted_results[n // 2]) / 2
        if n % 2 == 0
        else sorted_results[n // 2]
    )
    p25 = sorted_results[max(0, int(n * 0.25) - 1)]
    p75 = sorted_results[max(0, int(n * 0.75) - 1)]
    p90 = sorted_results[max(0, int(n * 0.90) - 1)]

    print(f"\n  Summary Statistics")
    print(f"  {'-'*58}")
    print(f"  Average turn  : {mean:.2f}")
    print(f"  Median turn   : {median:.1f}")
    print(f"  25th pct      : Turn {p25}")
    print(f"  75th pct      : Turn {p75}")
    print(f"  90th pct      : Turn {p90}")
    print(f"  Earliest      : Turn {sorted_results[0]}")
    print(f"  Latest cast   : Turn {sorted_results[-1]}")

    # --- Cumulative probability ---
    print(f"\n  Cumulative Probability")
    print(f"  {'-'*58}")
    cumulative = 0
    last_turn_with_data = sorted_results[-1]
    for turn in range(1, last_turn_with_data + 1):
        cumulative += turn_counts.get(turn, 0)
        pct = cumulative / total * 100
        if turn_counts.get(turn, 0) > 0:
            print(f"  By Turn {turn:2d}  : {pct:6.2f}%")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python commander_goldfish.py <decklist.txt>\n"
            "\n"
            "Decklist format:\n"
            "  Commander: Card Name\n"
            "  1 Card Name\n"
            "  4 Forest\n"
            "  ...\n",
            file=sys.stderr,
        )
        sys.exit(1)

    decklist_path = sys.argv[1]
    if not os.path.exists(decklist_path):
        print(f"Error: File not found: {decklist_path!r}", file=sys.stderr)
        sys.exit(1)

    print("\nCommander Goldfish Simulator", file=sys.stderr)
    print("=" * 40, file=sys.stderr)

    # --- Parse deck list ---
    print("Parsing decklist …", file=sys.stderr)
    try:
        commander_name, card_names = parse_decklist(decklist_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Commander : {commander_name}", file=sys.stderr)
    print(f"  Deck size : {len(card_names)} cards (+ commander in command zone)", file=sys.stderr)

    # --- Load / update card cache ---
    cache = _load_cache()

    print("Loading card data …", file=sys.stderr)
    try:
        commander, deck = build_deck(commander_name, card_names, cache)
    except (KeyError, ValueError) as exc:
        print(f"Error building deck: {exc}", file=sys.stderr)
        sys.exit(1)

    lands = [c for c in deck if c.is_land]
    rocks = [c for c in deck if c.is_mana_rock]
    dorks = [c for c in deck if c.is_mana_dork]
    print(f"  Commander CMC : {commander.cmc}", file=sys.stderr)
    print(f"  Lands         : {len(lands)}", file=sys.stderr)
    print(f"  Mana rocks    : {len(rocks)}", file=sys.stderr)
    print(f"  Mana dorks    : {len(dorks)}", file=sys.stderr)

    # --- Simulate ---
    print(f"\nRunning {NUM_SIMULATIONS:,} simulations …", file=sys.stderr)
    results = simulate_games(commander, deck, NUM_SIMULATIONS)

    # --- Report ---
    print_statistics(commander_name, commander.cmc, results)


if __name__ == "__main__":
    main()
