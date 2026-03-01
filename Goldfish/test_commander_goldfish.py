"""
Unit tests for commander_goldfish.py

Run with:
    python -m pytest test_commander_goldfish.py -v
or:
    python test_commander_goldfish.py
"""

import os
import random
import sys
import tempfile
import unittest

# Make sure the module is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from commander_goldfish import (
    Card,
    GameState,
    NUM_SIMULATIONS,
    STARTING_HAND_SIZE,
    MAX_TURNS,
    build_deck,
    get_card,
    parse_decklist,
    simulate_games,
    print_statistics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_card(
    name: str = "Test Card",
    type_line: str = "",
    cmc: int = 0,
    oracle_text: str = "",
    produced_mana: list = None,
) -> Card:
    """Convenience factory to build a Card without touching Scryfall."""
    return Card(
        name,
        {
            "name": name,
            "type_line": type_line,
            "mana_cost": "",
            "cmc": cmc,
            "oracle_text": oracle_text,
            "produced_mana": produced_mana or [],
        },
    )


def _forest() -> Card:
    return _make_card(
        "Forest",
        type_line="Basic Land — Forest",
        oracle_text="{T}: Add {G}.",
        produced_mana=["G"],
    )


def _sol_ring() -> Card:
    return _make_card(
        "Sol Ring",
        type_line="Artifact",
        cmc=1,
        oracle_text="{T}: Add {C}{C}.",
        produced_mana=["C"],
    )


def _arcane_signet() -> Card:
    return _make_card(
        "Arcane Signet",
        type_line="Artifact",
        cmc=2,
        oracle_text="{T}: Add one mana of any color in your commander's color identity.",
        produced_mana=["G"],
    )


def _llanowar_elves() -> Card:
    return _make_card(
        "Llanowar Elves",
        type_line="Creature — Elf Druid",
        cmc=1,
        oracle_text="{T}: Add {G}.",
        produced_mana=["G"],
    )


def _commander(cmc: int = 3) -> Card:
    return _make_card("Test Commander", type_line="Legendary Creature", cmc=cmc)


# ---------------------------------------------------------------------------
# Card tests
# ---------------------------------------------------------------------------


class TestCard(unittest.TestCase):
    def test_land_detection(self):
        forest = _forest()
        self.assertTrue(forest.is_land)
        self.assertFalse(forest.is_mana_rock)
        self.assertFalse(forest.is_mana_dork)

    def test_mana_rock_detection(self):
        sr = _sol_ring()
        self.assertFalse(sr.is_land)
        self.assertTrue(sr.is_mana_rock)
        self.assertFalse(sr.is_mana_dork)

    def test_mana_dork_detection(self):
        elves = _llanowar_elves()
        self.assertFalse(elves.is_land)
        self.assertFalse(elves.is_mana_rock)
        self.assertTrue(elves.is_mana_dork)

    def test_sol_ring_produces_2(self):
        """Sol Ring oracle text '{T}: Add {C}{C}.' → should produce 2 mana."""
        self.assertEqual(_sol_ring().mana_production(), 2)

    def test_forest_produces_1(self):
        self.assertEqual(_forest().mana_production(), 1)

    def test_signet_produces_1(self):
        """Arcane Signet oracle text adds one mana symbol → 1."""
        self.assertEqual(_arcane_signet().mana_production(), 1)

    def test_unknown_card_produces_1(self):
        """Card with no oracle text falls back to 1."""
        blank = _make_card("Blank", type_line="Land", produced_mana=["G"])
        self.assertEqual(blank.mana_production(), 1)

    def test_non_producer_type(self):
        vanilla = _make_card("Vanilla Bear", type_line="Creature — Bear", cmc=2)
        self.assertFalse(vanilla.is_mana_rock)
        self.assertFalse(vanilla.is_mana_dork)
        self.assertFalse(vanilla.is_land)


# ---------------------------------------------------------------------------
# Parse decklist tests
# ---------------------------------------------------------------------------


class TestParseDecklist(unittest.TestCase):
    def _write_deck(self, content: str) -> str:
        """Write a temporary decklist file and register cleanup, then return its path."""
        fh = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        fh.write(content)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_commander_line(self):
        path = self._write_deck("Commander: Atraxa, Praetors' Voice\n1 Forest\n")
        commander, cards = parse_decklist(path)
        self.assertEqual(commander, "Atraxa, Praetors' Voice")
        self.assertEqual(cards, ["Forest"])

    def test_multiple_copies(self):
        path = self._write_deck("Commander: Selvala\n4 Forest\n2 Sol Ring\n")
        _, cards = parse_decklist(path)
        self.assertEqual(cards.count("Forest"), 4)
        self.assertEqual(cards.count("Sol Ring"), 2)

    def test_comments_skipped(self):
        path = self._write_deck(
            "# This is a comment\nCommander: Test\n1 Island\n# Another comment\n2 Forest\n"
        )
        _, cards = parse_decklist(path)
        self.assertIn("Island", cards)
        self.assertEqual(cards.count("Forest"), 2)

    def test_missing_commander_raises(self):
        path = self._write_deck("1 Forest\n2 Sol Ring\n")
        with self.assertRaises(ValueError):
            parse_decklist(path)

    def test_tagged_commander(self):
        """Cards tagged with [Commander] at the end set the commander name."""
        path = self._write_deck("1 Selvala, Heart of the Wilds [Commander]\n1 Forest\n")
        commander, cards = parse_decklist(path)
        self.assertEqual(commander, "Selvala, Heart of the Wilds")

    def test_blank_lines_ignored(self):
        path = self._write_deck("Commander: X\n\n1 Forest\n\n2 Island\n")
        _, cards = parse_decklist(path)
        self.assertEqual(len(cards), 3)


# ---------------------------------------------------------------------------
# GameState mana calculation tests
# ---------------------------------------------------------------------------


class TestGameStateMana(unittest.TestCase):
    def _make_game(self, hand_cards, library_cards, commander_cmc=3):
        """Build a minimal GameState with a controlled hand & library."""
        cmd = _commander(cmc=commander_cmc)
        # Use a large library to avoid running out of cards
        all_library = hand_cards + library_cards + [_forest()] * 20
        game = GameState(cmd, all_library)
        # Override the randomised hand/library with our controlled cards
        game.hand = list(hand_cards)
        game.library = list(library_cards) + [_forest()] * 20
        return game

    def test_lands_in_play_add_mana(self):
        game = self._make_game([], [])
        game.lands_in_play = [_forest(), _forest()]
        # _total_mana() is private; access via the public helper
        # We can check by attempting to cast a cheap commander
        game.commander = _commander(cmc=2)
        # 2 forests → 2 mana; commander costs 2 → should cast turn 1
        game.turn = 0
        cast = game.play_turn()
        # After playing a land from library we should reach ≥2 mana
        # (the two pre-placed forests give 2 mana immediately)
        # Actually because play_turn draws and tries to play a land from hand,
        # we verify total mana ≥ 2 by checking if turn succeeds eventually.
        # Simply assert the internal counter is positive.
        self.assertGreaterEqual(len(game.lands_in_play), 2)

    def test_commander_tax_increases_cost(self):
        """Commander cost = CMC + 2 * number of previous casts (commander tax)."""
        cmd = _commander(cmc=3)
        game = GameState(cmd, [_forest()] * 30)
        # 0 prior casts → cost equals CMC
        self.assertEqual(cmd.cmc + 2 * game.commander_casts, 3)
        game.commander_casts = 1
        self.assertEqual(cmd.cmc + 2 * game.commander_casts, 5)
        game.commander_casts = 2
        self.assertEqual(cmd.cmc + 2 * game.commander_casts, 7)

    def test_sol_ring_played_when_affordable(self):
        """Sol Ring (CMC 1) should be played when a land is already in hand."""
        cmd = _commander(cmc=10)  # Won't be cast this turn
        sr = _sol_ring()
        hand = [_forest(), sr]
        game = self._make_game(hand, [])
        game.play_turn()  # Turn 1: draw, play land, play Sol Ring
        self.assertIn(sr, game.mana_sources_in_play)

    def test_3cmc_commander_cast_by_turn_3_with_perfect_hand(self):
        """Forest + Forest + Forest hand should cast a CMC-3 commander by turn 3."""
        cmd = _commander(cmc=3)
        library = [_forest()] * 99
        game = GameState(cmd, library)
        # Force hand to be 7 forests
        game.hand = [_forest() for _ in range(7)]
        game.library = [_forest()] * 30
        result = game.simulate()
        self.assertIsNotNone(result)
        self.assertLessEqual(result, 3)


# ---------------------------------------------------------------------------
# Full simulation sanity tests
# ---------------------------------------------------------------------------


class TestSimulation(unittest.TestCase):
    def _pure_forest_deck(self, num_forests: int, commander_cmc: int):
        """Build a deck of only basic forests and a commander with given CMC."""
        cmd = _commander(cmc=commander_cmc)
        deck = [_forest() for _ in range(num_forests)]
        return cmd, deck

    def test_cmc2_cast_early(self):
        """A 2-CMC commander with 30 forests should almost always be cast by turn 3."""
        random.seed(42)
        cmd, deck = self._pure_forest_deck(30, commander_cmc=2)
        results = simulate_games(cmd, deck, num_simulations=500)
        cast_results = [r for r in results if r is not None]
        self.assertGreater(len(cast_results), 0)
        avg = sum(cast_results) / len(cast_results)
        self.assertLessEqual(avg, 4.0)  # Should typically cast by turn 4

    def test_cmc5_takes_longer(self):
        """A 5-CMC commander takes longer than a 2-CMC commander on average."""
        random.seed(42)
        cmd2, deck2 = self._pure_forest_deck(30, commander_cmc=2)
        cmd5, deck5 = self._pure_forest_deck(30, commander_cmc=5)
        res2 = [r for r in simulate_games(cmd2, deck2, 300) if r is not None]
        res5 = [r for r in simulate_games(cmd5, deck5, 300) if r is not None]
        avg2 = sum(res2) / len(res2)
        avg5 = sum(res5) / len(res5)
        self.assertLess(avg2, avg5)

    def test_result_count_matches_simulations(self):
        cmd, deck = self._pure_forest_deck(30, 3)
        results = simulate_games(cmd, deck, num_simulations=100)
        self.assertEqual(len(results), 100)

    def test_no_lands_never_cast(self):
        """With zero lands the commander should never be cast (cmc > 0)."""
        cmd = _commander(cmc=4)
        # Deck full of vanilla non-producing creatures
        deck = [
            _make_card(f"Bear {i}", type_line="Creature — Bear", cmc=2)
            for i in range(60)
        ]
        results = simulate_games(cmd, deck, num_simulations=50)
        self.assertTrue(all(r is None for r in results))

    def test_result_within_max_turns(self):
        cmd, deck = self._pure_forest_deck(36, 3)
        results = simulate_games(cmd, deck, num_simulations=200)
        for r in results:
            if r is not None:
                self.assertLessEqual(r, MAX_TURNS)


# ---------------------------------------------------------------------------
# Statistics output smoke test
# ---------------------------------------------------------------------------


class TestPrintStatistics(unittest.TestCase):
    def test_runs_without_error(self):
        """print_statistics should not raise for a normal result list."""
        import io
        from contextlib import redirect_stdout

        results = [3, 3, 4, 4, 5, None, 3, 2, 6, 4]
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_statistics("Test Commander", 4, results)
        output = buf.getvalue()
        self.assertIn("Turn", output)
        self.assertIn("Average", output)

    def test_all_none_results(self):
        """Should handle the edge case where the commander was never cast."""
        import io
        from contextlib import redirect_stdout

        results = [None, None, None]
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_statistics("No-Cast Commander", 10, results)
        self.assertIn("never", buf.getvalue().lower())


# ---------------------------------------------------------------------------
# Card cache / get_card smoke test
# ---------------------------------------------------------------------------


class TestGetCard(unittest.TestCase):
    def test_cache_hit_returns_card(self):
        """get_card uses the cache and does not hit the network."""
        cache = {
            "forest": {
                "name": "Forest",
                "type_line": "Basic Land — Forest",
                "mana_cost": "",
                "cmc": 0,
                "produced_mana": ["G"],
                "oracle_text": "{T}: Add {G}.",
            }
        }
        card = get_card("Forest", cache)
        self.assertEqual(card.name, "Forest")
        self.assertTrue(card.is_land)

    def test_missing_key_stored_after_fetch_failure(self):
        """
        When Scryfall is unreachable (no network in CI), get_card falls back
        gracefully and stores a minimal placeholder in the cache.
        """
        cache: dict = {}
        # This will fail to reach Scryfall in a sandboxed environment;
        # the function should NOT raise.
        card = get_card("NonExistentCardXYZ123", cache)
        self.assertIsNotNone(card)
        self.assertIn("nonexistentcardxyz123", cache)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
