"""
Headless AI-vs-AI benchmark runner for Halma.

Run multiple self-play games to compare search settings (depth, pruning, time).
Example:
    python arena.py --games 3 --size 8 --time-limit 10 --depth-g 3 --depth-r 3

Use --no-alpha-beta to turn pruning off for both, or set per-color flags.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# Package/script import shim
try:  # pragma: no cover
    from .ai import HalmaAI, Move  # type: ignore
    from .board import HalmaBoard  # type: ignore
except ImportError:  # pragma: no cover
    from halma.ai import HalmaAI, Move  # type: ignore
    from halma.board import HalmaBoard  # type: ignore


@dataclass
class GameResult:
    winner: Optional[str]
    moves: int
    duration: float
    nodes_g: int
    nodes_r: int
    started: str


def play_game(
    size: int,
    time_limit_g: float,
    time_limit_r: float,
    depth_g: int,
    depth_r: int,
    use_ab_g: bool,
    use_ab_r: bool,
    max_moves: int,
    start_player: str,
) -> GameResult:
    # Headless self-play to sanity check search speed/quality without the GUI.
    board = HalmaBoard(size)
    board.setup_starting_positions()

    ai_g = HalmaAI(max_depth=depth_g, use_alpha_beta=use_ab_g)
    ai_r = HalmaAI(max_depth=depth_r, use_alpha_beta=use_ab_r)

    current = start_player
    move_count = 0
    start_time = time.monotonic()
    nodes_g = 0
    nodes_r = 0

    while move_count < max_moves:
        if board.has_won(current):
            break

        ai = ai_g if current == "G" else ai_r
        time_budget = time_limit_g if current == "G" else time_limit_r
        move, stats = ai.choose_move(
            board,
            current,
            time_limit=time_budget,
            max_depth=ai.max_depth,
            use_alpha_beta=ai.use_alpha_beta,
        )
        if current == "G":
            nodes_g += stats.nodes_evaluated
        else:
            nodes_r += stats.nodes_evaluated

        if move is None:
            # No legal moves or timeout: forfeit.
            winner = "R" if current == "G" else "G"
            return GameResult(
                winner=winner,
                moves=move_count,
                duration=time.monotonic() - start_time,
                nodes_g=nodes_g,
                nodes_r=nodes_r,
                started=start_player,
            )

        board.move_piece(current, move.start, move.dest)
        move_count += 1

        if board.has_won(current):
            return GameResult(
                winner=current,
                moves=move_count,
                duration=time.monotonic() - start_time,
                nodes_g=nodes_g,
                nodes_r=nodes_r,
                started=start_player,
            )

        current = "R" if current == "G" else "G"

    return GameResult(
        winner=None,
        moves=move_count,
        duration=time.monotonic() - start_time,
        nodes_g=nodes_g,
        nodes_r=nodes_r,
        started=start_player,
    )


def summarize(results: list[GameResult]) -> None:
    def safe_mean(vals):
        return statistics.mean(vals) if vals else 0.0

    games = len(results)
    wins_g = sum(1 for r in results if r.winner == "G")
    wins_r = sum(1 for r in results if r.winner == "R")
    draws = sum(1 for r in results if r.winner is None)
    moves = [r.moves for r in results]
    times = [r.duration for r in results]
    nodes_g = [r.nodes_g for r in results]
    nodes_r = [r.nodes_r for r in results]

    print(f"Games: {games} | Green wins: {wins_g} | Red wins: {wins_r} | Draws/forfeits: {draws}")
    print(f"Avg moves: {safe_mean(moves):.1f} | Avg time: {safe_mean(times):.2f}s")
    print(f"Avg nodes (G): {safe_mean(nodes_g):.0f} | Avg nodes (R): {safe_mean(nodes_r):.0f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless AI-vs-AI Halma benchmark.")
    p.add_argument("--size", type=int, default=8, choices=(8, 10, 16), help="Board size (default 8).")
    p.add_argument("--time-limit", type=float, default=10.0, help="Per-move think time (sec, default for both).")
    p.add_argument("--time-limit-g", type=float, help="Per-move think time for Green (overrides --time-limit).")
    p.add_argument("--time-limit-r", type=float, help="Per-move think time for Red (overrides --time-limit).")
    p.add_argument("--depth-g", type=int, default=3, help="Max depth for Green AI.")
    p.add_argument("--depth-r", type=int, default=3, help="Max depth for Red AI.")
    p.add_argument("--no-alpha-beta", action="store_true", help="Disable alpha-beta for both.")
    p.add_argument("--no-alpha-beta-g", action="store_true", help="Disable alpha-beta for Green only.")
    p.add_argument("--no-alpha-beta-r", action="store_true", help="Disable alpha-beta for Red only.")
    p.add_argument("--games", type=int, default=1, help="Number of games to run.")
    p.add_argument("--max-moves", type=int, default=400, help="Maximum plies before declaring a draw.")
    p.add_argument(
        "--start-player",
        type=str,
        default="G",
        choices=("G", "R", "alternate"),
        help="Who moves first (G/R), or alternate each game.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    time_limit_g = args.time_limit_g if args.time_limit_g is not None else args.time_limit
    time_limit_r = args.time_limit_r if args.time_limit_r is not None else args.time_limit
    use_ab_g = not (args.no_alpha_beta or args.no_alpha_beta_g)
    use_ab_r = not (args.no_alpha_beta or args.no_alpha_beta_r)

    print(
        f"Config — size={args.size}, "
        f"G(depth={args.depth_g}, time={time_limit_g}, ab={use_ab_g}) vs "
        f"R(depth={args.depth_r}, time={time_limit_r}, ab={use_ab_r}), "
        f"start={args.start_player}, games={args.games}"
    )

    start_g = start_r = 0
    win_start_g = win_start_r = 0

    results: list[GameResult] = []
    for game in range(1, args.games + 1):
        print(f"=== Game {game} ===")
        start_player = (
            "G"
            if args.start_player == "G"
            else "R"
            if args.start_player == "R"
            else ("G" if game % 2 == 1 else "R")
        )
        if start_player == "G":
            start_g += 1
        else:
            start_r += 1
        res = play_game(
            size=args.size,
            time_limit_g=time_limit_g,
            time_limit_r=time_limit_r,
            depth_g=args.depth_g,
            depth_r=args.depth_r,
            use_ab_g=use_ab_g,
            use_ab_r=use_ab_r,
            max_moves=args.max_moves,
            start_player=start_player,
        )
        results.append(res)
        outcome = res.winner if res.winner else "draw/forfeit"
        if res.winner == "G" and start_player == "G":
            win_start_g += 1
        elif res.winner == "R" and start_player == "R":
            win_start_r += 1
        print(
            f"Result: {outcome}, start={start_player}, moves={res.moves}, "
            f"time={res.duration:.2f}s, nodes G={res.nodes_g}, nodes R={res.nodes_r}"
        )
    print("\n=== Summary ===")
    summarize(results)
    if start_g or start_r:
        print(
            f"Starts — G: {start_g} (wins as starter: {win_start_g}), "
            f"R: {start_r} (wins as starter: {win_start_r})"
        )


if __name__ == "__main__":
    main()
