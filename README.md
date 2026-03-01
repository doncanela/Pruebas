# Commander Goldfish Simulator

Simulates **10,000 goldfish starts** to see on which turn you can play your
commander.

A *goldfish* game means playing with no opponent – the simulation focuses
purely on your own mana development to find out how quickly the commander
comes down.

---

## Requirements

* Python 3.9 or newer (uses only the standard library)

---

## Usage

```bash
python commander_goldfish.py <decklist.txt>
```

### Decklist format

Plain text, one card per line:

```
Commander: Selvala, Heart of the Wilds

# Mana rocks
1 Sol Ring
1 Arcane Signet

# Lands
36 Forest

# Other spells
1 Llanowar Elves
1 Craterhoof Behemoth
...
```

Rules:
- **`Commander: <name>`** – required line that identifies the commander.
- **`<count> <name>`** – standard "count name" format (Moxfield / EDHREC).
- Lines starting with `#` are comments and are ignored.
- The commander is excluded from the library (it starts in the command zone).

An example deck is provided in [`example_deck.txt`](example_deck.txt).

---

## How it works

Each simulated game:
1. Shuffles the 99-card library and deals a 7-card opening hand (no mulligans).
2. Each turn: draws a card, plays a land if available, then plays any
   affordable mana rocks / mana dorks (cheapest first), accurately tracking
   mana spent and gained.
3. Checks whether the remaining mana pool covers the commander's cost
   (including **commander tax** of +2 per previous cast).
4. Records the turn the commander was cast, or `None` if it wasn't cast within
   20 turns.

After 10,000 games the tool prints a turn-distribution bar chart, summary
statistics (average, median, percentiles), and a cumulative probability table.

Card data (type line, CMC, oracle text) is fetched automatically from the
[Scryfall API](https://scryfall.com/docs/api) on the first run and cached
locally in `.card_cache.json` for subsequent runs.

---

## Example output

```
==============================================================
  Commander Goldfish Simulation Results
==============================================================
  Commander : Selvala, Heart of the Wilds
  CMC       : 3
  Games     : 10,000
==============================================================

  Turn Distribution
  ----------------------------------------------------------
  Turn  2:  4,832  ( 48.3%)  ████████████████████████████████████████
  Turn  3:  3,105  ( 31.1%)  █████████████████████████
  Turn  4:  1,612  ( 16.1%)  █████████████
  ...

  Summary Statistics
  ----------------------------------------------------------
  Average turn  : 2.73
  Median turn   : 3.0
  25th pct      : Turn 2
  75th pct      : Turn 3
  90th pct      : Turn 4
  ...
```

---

## Running the tests

```bash
python -m pytest test_commander_goldfish.py -v
```