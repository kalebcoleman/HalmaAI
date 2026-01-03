# Halma AI

AI-enabled Halma player with a Tkinter GUI and a headless arena for self-play.

## Features
- Minimax with optional alpha-beta pruning
- Iterative deepening with a per-move time budget
- Move ordering to improve pruning
- Heuristic evaluation that rewards progress toward the goal camp
- GUI for human vs AI or AI vs AI
- Headless arena for benchmarking search settings

## Requirements
- Python 3.13+ (Tkinter included with most standard installs)

## Quick start (GUI)
From this folder:

```bash
python __main__.py
```

Common options:

```bash
python __main__.py --size 8 --ai-color red --search-depth 3 --time-limit 20
```

## Headless arena (AI vs AI)

```bash
python arena.py --games 5 --start-player alternate --depth-g 3 --depth-r 4 --time-limit 10
```

## Project layout
- `ai.py` — minimax/alpha-beta logic and evaluation
- `board.py` — rules, state, and move validation
- `gui.py` — Tkinter interface
- `arena.py` — headless benchmark runner
- `__main__.py` — CLI entry point for the GUI
- `tuning_notes.md` — example benchmark commands

## Notes
Board sizes supported: 8x8, 10x10, and 16x16.
