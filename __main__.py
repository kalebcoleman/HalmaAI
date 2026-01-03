"""
Command-line entry point for the Halma Part 1 manual game.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import messagebox

# Support running as both a package and a script.
try:  # pragma: no cover - import shim
    from .board import HalmaBoard  # type: ignore
    from .gui import HalmaGUI  # type: ignore
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from board import HalmaBoard  # type: ignore
    from gui import HalmaGUI  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Halma manual/AI match runner.")
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        choices=(8, 10, 16),
        help="Board size to display (default 8).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20.0,
        help="Per-move time limit in seconds (default 20).",
    )
    parser.add_argument(
        "--human-color",
        type=str,
        default="green",
        choices=("green", "red"),
        help="Highlight which color you intend to sit at (display purposes).",
    )
    parser.add_argument(
        "--ai-color",
        type=str,
        default="red",
        choices=("none", "green", "red", "both"),
        help="Which side(s) the AI controls (default red).",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=3,
        help="Maximum ply depth for minimax search (default 3).",
    )
    parser.add_argument(
        "--ai-think-time",
        type=float,
        default=None,
        help="Optional override for AI think time budget in seconds (defaults to time-limit).",
    )
    parser.add_argument(
        "--no-alpha-beta",
        action="store_true",
        help="Disable alpha-beta pruning for debugging or comparison runs.",
    )
    parser.add_argument(
        "--board-file",
        type=Path,
        help="Optional path to a board layout text file.",
    )
    return parser


def run_game(
    *,
    size: int,
    time_limit: float,
    human_color: str,
    ai_color: str,
    search_depth: int,
    ai_think_time: Optional[float],
    use_alpha_beta: bool,
    board_file: Optional[Path],
) -> None:
    # Keep setup simple: load board file if given, otherwise start fresh.
    board = HalmaBoard(size)
    if board_file:
        board.load_from_file(board_file)
    else:
        board.setup_starting_positions()

    root = tk.Tk()
    root.title(f"Halma Match ({size}x{size})")
    HalmaGUI(
        root,
        board,
        time_limit=time_limit,
        human_color=human_color,
        ai_color=ai_color,
        search_depth=search_depth,
        ai_time_limit=ai_think_time,
        use_alpha_beta=use_alpha_beta,
    )
    root.mainloop()


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_game(
            size=args.size,
            time_limit=args.time_limit,
            human_color=args.human_color,
            ai_color=args.ai_color,
            search_depth=args.search_depth,
            ai_think_time=args.ai_think_time,
            use_alpha_beta=not args.no_alpha_beta,
            board_file=args.board_file,
        )
    except ValueError as exc:
        err_root = tk.Tk()
        err_root.withdraw()
        messagebox.showerror("Configuration error", str(exc))
        err_root.destroy()


if __name__ == "__main__":
    main()
