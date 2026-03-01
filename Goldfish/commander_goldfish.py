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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


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

        # Cards whose mana can only be spent on abilities (e.g. Cryptic
        # Trilobite) cannot help cast the commander, so exclude them.
        _restricted_mana: bool = bool(
            re.search(r"spend this mana only", self.oracle_text, re.IGNORECASE)
        )

        # X-cost creatures / artifacts (e.g. Hangarback Walker, Stonecoil
        # Serpent) have CMC 0 in Scryfall but casting them for X=0 means a
        # 0/0 creature that dies immediately.  Don't treat them as free plays.
        _x_cost_zero: bool = "{X}" in self.mana_cost and self.cmc == 0

        self.is_mana_rock: bool = (
            "Artifact" in self.type_line
            and not self.is_land
            and bool(self.produced_mana)
            and not _restricted_mana
            and not _x_cost_zero
        )
        self.is_mana_dork: bool = (
            "Creature" in self.type_line
            and not self.is_land
            and bool(self.produced_mana)
            and not _restricted_mana
            and not _x_cost_zero
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
        headers={
            "User-Agent": "CommanderGoldfishSimulator/1.0 (educational tool)",
            "Accept": "application/json",
        },
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

# Sections whose cards should be excluded from the library
_EXCLUDED_SECTIONS = {"maybeboard", "tokens & extras", "tokens"}


def parse_decklist(filepath: str) -> Tuple[str, List[str]]:
    """
    Parse a plain-text decklist.  Supports two formats:

    **Classic format** (Moxfield / EDHREC / manual):
        Commander: Card Name
        # Category (comment)
        1 Card Name

    **Archidekt export format**:
        Commander
        1 Card Name
        Anthem
        1 Another Card
        Tokens & Extras
        1 Token Name        <-- excluded

    Returns:
        (commander_name, flat_list_of_card_names)

    Cards under "Maybeboard", "Tokens & Extras", or "Tokens" sections are
    excluded automatically.
    """
    commander_name: Optional[str] = None
    cards: List[str] = []
    current_section: Optional[str] = None   # Tracks the active category header
    in_commander_section: bool = False       # True when under an "Commander" header

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Classic format: "Commander: Card Name"
            if line.lower().startswith("commander:"):
                rest = line.split(":", 1)[1].strip()
                if rest:  # "Commander: Card Name" (classic)
                    commander_name = rest
                    in_commander_section = False
                else:     # Bare "Commander:" with no name — treat like header
                    current_section = "commander"
                    in_commander_section = True
                continue

            # Try to match a card line: "<count> <card name>" with optional tag
            match = re.match(r"^(\d+)\s+(.+?)(\s+\[Commander\])?\s*$", line, re.IGNORECASE)

            if match:
                count = int(match.group(1))
                card_name = match.group(2).strip()
                is_tagged_commander = bool(match.group(3))

                # Card inside the Commander section → this is the commander
                if in_commander_section and commander_name is None:
                    commander_name = card_name
                    in_commander_section = False
                    continue

                if is_tagged_commander and commander_name is None:
                    commander_name = card_name

                # Skip cards in excluded sections
                if current_section and current_section.lower() in _EXCLUDED_SECTIONS:
                    continue

                cards.extend([card_name] * count)
            else:
                # Line is not a card entry → treat as a category header
                current_section = line
                in_commander_section = line.lower() == "commander"

    if not commander_name:
        raise ValueError(
            "No commander found in the decklist.\n"
            "Add a line like:  Commander: Your Commander Name\n"
            "Or use an Archidekt-style 'Commander' section header."
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
# Visual report (matplotlib)
# ---------------------------------------------------------------------------

def generate_report_image(
    commander_name: str,
    commander_cmc: int,
    results: List[Optional[int]],
    deck_size: int,
    num_lands: int,
    num_rocks: int,
    num_dorks: int,
    output_path: str,
) -> None:
    """
    Generate a visual PNG report with turn-distribution bar chart,
    cumulative-probability line chart, and a summary statistics panel.
    """
    total = len(results)
    cast_results = [r for r in results if r is not None]
    never_cast = total - len(cast_results)
    turn_counts: Counter = Counter(cast_results)

    if not cast_results:
        print("  No cast data to graph — commander was never cast.", file=sys.stderr)
        return

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

    # Build data arrays
    last_turn = sorted_results[-1]
    turns = list(range(1, last_turn + 1))
    counts = [turn_counts.get(t, 0) for t in turns]
    pcts = [c / total * 100 for c in counts]

    cumulative = []
    running = 0
    for c in counts:
        running += c
        cumulative.append(running / total * 100)

    # ---- Figure layout: 2×2 grid ----
    fig = plt.figure(figsize=(14, 10), facecolor="#1e1e2e")
    fig.suptitle(
        f"Commander Goldfish — {commander_name}",
        fontsize=18, fontweight="bold", color="white", y=0.97,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.95, top=0.90, bottom=0.06)

    ax_bar = fig.add_subplot(gs[0, :])
    ax_cum = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])

    # Shared axis styling helper
    def style_ax(ax, title):
        ax.set_facecolor("#2a2a3d")
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=10)
        ax.tick_params(colors="#cccccc", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    # ---- 1. Turn Distribution bar chart ----
    colors = ["#66bbff" if t != round(median) else "#ffcc44" for t in turns]
    bars = ax_bar.bar(turns, pcts, color=colors, edgecolor="#1e1e2e", linewidth=0.5)
    style_ax(ax_bar, "Turn Distribution (%)")
    ax_bar.set_xlabel("Turn")
    ax_bar.set_ylabel("Games (%)")
    ax_bar.set_xticks(turns)
    ax_bar.axvline(mean, color="#ff6666", linestyle="--", linewidth=1.2, label=f"Mean {mean:.2f}")
    ax_bar.axvline(median, color="#ffcc44", linestyle="-", linewidth=1.5, label=f"Median {median:.0f}")
    ax_bar.legend(loc="upper right", fontsize=9, facecolor="#2a2a3d",
                  edgecolor="#444466", labelcolor="white")
    # Value labels on bars
    for bar, pct in zip(bars, pcts):
        if pct >= 1.0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#cccccc")
    if never_cast:
        ax_bar.annotate(
            f"Never cast: {never_cast} ({never_cast / total * 100:.1f}%)",
            xy=(0.99, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=9, color="#ff8888",
            bbox=dict(boxstyle="round,pad=0.3", fc="#2a2a3d", ec="#ff8888", alpha=0.9),
        )

    # ---- 2. Cumulative Probability line chart ----
    ax_cum.plot(turns, cumulative, color="#66ffaa", linewidth=2.2, marker="o",
                markersize=4, markerfacecolor="#44dd88")
    ax_cum.fill_between(turns, cumulative, alpha=0.15, color="#66ffaa")
    style_ax(ax_cum, "Cumulative Probability (%)")
    ax_cum.set_xlabel("Turn")
    ax_cum.set_ylabel("Cumulative %")
    ax_cum.set_xticks(turns)
    ax_cum.set_ylim(0, 105)
    ax_cum.axhline(50, color="#888888", linestyle=":", linewidth=0.8)
    ax_cum.axhline(90, color="#888888", linestyle=":", linewidth=0.8)
    ax_cum.text(turns[-1], 51, "50%", fontsize=7.5, color="#aaaaaa", ha="right")
    ax_cum.text(turns[-1], 91, "90%", fontsize=7.5, color="#aaaaaa", ha="right")
    # Mark the 50% and 90% crossing turns
    for threshold, clr in [(50, "#ffcc44"), (90, "#ff6666")]:
        for i, val in enumerate(cumulative):
            if val >= threshold:
                ax_cum.plot(turns[i], val, "D", color=clr, markersize=7, zorder=5)
                ax_cum.annotate(
                    f"Turn {turns[i]}", (turns[i], val),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=clr, fontweight="bold",
                )
                break

    # ---- 3. Summary statistics info panel ----
    ax_info.set_facecolor("#2a2a3d")
    ax_info.set_title("Summary", fontsize=13, fontweight="bold", color="white", pad=10)
    ax_info.axis("off")

    info_lines = [
        ("Commander", commander_name),
        ("CMC", str(commander_cmc)),
        ("Deck size", f"{deck_size} cards"),
        ("Lands", str(num_lands)),
        ("Mana rocks", str(num_rocks)),
        ("Mana dorks", str(num_dorks)),
        ("", ""),
        ("Simulations", f"{total:,}"),
        ("Average turn", f"{mean:.2f}"),
        ("Median turn", f"{median:.1f}"),
        ("25th pct", f"Turn {p25}"),
        ("75th pct", f"Turn {p75}"),
        ("90th pct", f"Turn {p90}"),
        ("Earliest", f"Turn {sorted_results[0]}"),
        ("Latest cast", f"Turn {sorted_results[-1]}"),
    ]
    if never_cast:
        info_lines.append(("Never cast", f"{never_cast} ({never_cast / total * 100:.2f}%)"))

    y_pos = 0.95
    for label, value in info_lines:
        if not label and not value:
            y_pos -= 0.03
            continue
        ax_info.text(0.05, y_pos, f"{label}:", transform=ax_info.transAxes,
                     fontsize=10, color="#999999", fontweight="bold", va="top")
        ax_info.text(0.55, y_pos, value, transform=ax_info.transAxes,
                     fontsize=10, color="white", va="top")
        y_pos -= 0.065

    # ---- Save ----
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved to: {output_path}", file=sys.stderr)


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

    # --- Resolve paths: decklists/ for input, output/ for results ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    decklists_dir = os.path.join(script_dir, "decklists")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(decklists_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw_path = sys.argv[1]
    # Allow both bare filenames (looked up in decklists/) and full paths
    if os.path.isabs(raw_path) or os.path.exists(raw_path):
        decklist_path = raw_path
    else:
        decklist_path = os.path.join(decklists_dir, raw_path)

    if not os.path.exists(decklist_path):
        print(f"Error: File not found: {decklist_path!r}", file=sys.stderr)
        print(f"  Looked in: {decklists_dir}", file=sys.stderr)
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

    # --- Visual report ---
    deck_stem = Path(decklist_path).stem            # e.g. "iron_spider, stark upgrade"
    safe_name = re.sub(r'[^\w\s-]', '', deck_stem)  # strip special chars
    safe_name = re.sub(r'\s+', '_', safe_name)       # spaces → underscores
    output_file = os.path.join(
        output_dir,
        f"output_{safe_name}_goldfish_test.png",
    )

    print(f"\nGenerating visual report …", file=sys.stderr)
    generate_report_image(
        commander_name=commander_name,
        commander_cmc=commander.cmc,
        results=results,
        deck_size=len(card_names),
        num_lands=len(lands),
        num_rocks=len(rocks),
        num_dorks=len(dorks),
        output_path=output_file,
    )


if __name__ == "__main__":
    main()
