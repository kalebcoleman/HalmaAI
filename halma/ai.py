"""
AI agent and search logic for Halma Part 2.

Implements a depth-limited minimax search with optional alpha-beta pruning,
iterative deepening with a wall-clock budget, and a utility function that
rewards forward progress into the goal camp while penalizing pieces lingering
in the home camp.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Support running as both a package and a script.
try:  # pragma: no cover - import shim
    from .board import Coord, HalmaBoard  # type: ignore
except ImportError:  # pragma: no cover
    from halma.board import Coord, HalmaBoard  # type: ignore


@dataclass(frozen=True)
class Move:
    start: Coord
    dest: Coord
    kind: str  # "step" or "jump"


@dataclass
class SearchStats:
    depth_reached: int
    nodes_evaluated: int
    prunes_by_depth: Dict[int, int]
    elapsed: float
    best_value: float


class SearchTimeout(Exception):
    """Raised internally when the allotted think-time is exceeded."""


class HalmaAI:
    """Depth-limited minimax/alpha-beta search for Halma."""

    def __init__(self, max_depth: int = 3, use_alpha_beta: bool = True) -> None:
        # Keep knobs tiny so you can tweak depth/pruning without spelunking.
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta

        self._deadline: float = 0.0
        self._root_player: str = "G"
        self._nodes: int = 0
        self._prunes: Dict[int, int] = {}

    # ------------------------------------------------------------------ Public entry point

    def choose_move(
        self,
        board: HalmaBoard,
        player: str,
        *,
        time_limit: float,
        max_depth: Optional[int] = None,
        use_alpha_beta: Optional[bool] = None,
    ) -> Tuple[Optional[Move], SearchStats]:
        """
        Select a move for ``player`` from ``board`` using minimax.

        The search uses iterative deepening so it can return the best fully
        explored depth when the clock expires.
        """

        depth_cap = max_depth or self.max_depth
        pruning = self.use_alpha_beta if use_alpha_beta is None else use_alpha_beta

        self._root_player = player
        self._nodes = 0
        self._prunes = {}
        self._last_best: Optional[Move] = None

        # Keep a small safety cushion so we return something instead of timing out.
        safety_margin = max(0.05, time_limit * 0.05)
        self._deadline = time.monotonic() + max(0.01, time_limit - safety_margin)

        best_move: Optional[Move] = None
        best_value = -math.inf
        depth_reached = 0

        start_time = time.monotonic()

        # Iterative deepening: try shallow first, keep the best full depth before time runs out.
        for depth in range(1, depth_cap + 1):
            try:
                value, move = self._minimax(
                    board,
                    depth_remaining=depth,
                    maximizing=True,
                    alpha=-math.inf,
                    beta=math.inf,
                    player_to_move=player,
                    use_alpha_beta=pruning,
                )
                if move is not None:
                    best_move = move
                    self._last_best = move
                best_value = value
                depth_reached = depth
            except SearchTimeout:
                break

        elapsed = time.monotonic() - start_time
        stats = SearchStats(
            depth_reached=depth_reached,
            nodes_evaluated=self._nodes,
            prunes_by_depth=dict(sorted(self._prunes.items())),
            elapsed=elapsed,
            best_value=best_value,
        )
        return best_move, stats

    def evaluate_board(self, board: HalmaBoard, player: str) -> float:
        """
        Public helper to get the utility value of a board for `player`.
        """
        self._root_player = player
        return self._evaluate(board)

    # ------------------------------------------------------------------ Minimax + helpers

    def _minimax(
        self,
        board: HalmaBoard,
        *,
        depth_remaining: int,
        maximizing: bool,
        alpha: float,
        beta: float,
        player_to_move: str,
        use_alpha_beta: bool,
    ) -> Tuple[float, Optional[Move]]:
        self._check_time()
        self._nodes += 1

        opponent = self._opponent(player_to_move)
        # Early outs: somebody already won or we've hit depth.
        if board.has_won(self._root_player):
            return math.inf, None
        if board.has_won(self._opponent(self._root_player)):
            return -math.inf, None
        if depth_remaining == 0:
            return self._evaluate(board), None

        legal_moves = list(self._generate_moves(board, player_to_move))
        # Try the previously best move first to improve pruning (principal variation hint).
        if self._last_best is not None:
            try:
                idx = legal_moves.index(self._last_best)
                legal_moves[0], legal_moves[idx] = legal_moves[idx], legal_moves[0]
            except ValueError:
                pass  # last best not legal here
        if not legal_moves:
            # No legal moves — treat as a loss for the side to move.
            return (-math.inf if maximizing else math.inf), None

        # Move ordering: reward forward progress to help pruning do its job.
        legal_moves.sort(key=lambda m: self._move_progress(board, player_to_move, m), reverse=True)

        best_move: Optional[Move] = None
        if maximizing:
            best_value = -math.inf
            for move in legal_moves:
                self._apply_move(board, player_to_move, move)
                val, _ = self._minimax(
                    board,
                    depth_remaining=depth_remaining - 1,
                    maximizing=False,
                    alpha=alpha,
                    beta=beta,
                    player_to_move=opponent,
                    use_alpha_beta=use_alpha_beta,
                )
                self._undo_move(board, player_to_move, move)
                if val > best_value:
                    best_value = val
                    best_move = move
                if use_alpha_beta:
                    alpha = max(alpha, best_value)
                    if beta <= alpha:
                        self._record_prune(depth_remaining)
                        break
            return best_value, best_move
        else:
            best_value = math.inf
            for move in legal_moves:
                self._apply_move(board, player_to_move, move)
                val, _ = self._minimax(
                    board,
                    depth_remaining=depth_remaining - 1,
                    maximizing=True,
                    alpha=alpha,
                    beta=beta,
                    player_to_move=opponent,
                    use_alpha_beta=use_alpha_beta,
                )
                self._undo_move(board, player_to_move, move)
                if val < best_value:
                    best_value = val
                    best_move = move
                if use_alpha_beta:
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        self._record_prune(depth_remaining)
                        break
            return best_value, best_move

    def _generate_moves(self, board: HalmaBoard, player: str) -> Iterable[Move]:
        for coord in board.pieces[player]:
            for dest in board.legal_steps(coord, player):
                yield Move(coord, dest, "step")
            for dest in board.legal_jumps(coord, player, coord, {coord}):
                yield Move(coord, dest, "jump")

    def _evaluate(self, board: HalmaBoard) -> float:
        """
        Distance-driven heuristic scaled for clearer utility values.

        - Uses precomputed distance_cache to nearest goal square.
        - Adds a stronger bonus for pieces already in goal and a heavier
          penalty for lingering in home to break symmetry early.
        - Emphasizes stragglers by adding the largest three distances again.
        - Returns opponent minus mine so higher is better for the side to move.
        """

        def side_score(player: str) -> float:
            distances: List[float] = []
            score = 0.0
            for (r, c) in board.pieces[player]:
                if (r, c) in board.goal_camps[player]:
                    score -= 12.0  # bigger reward for being in goal
                    continue
                dist = board.distance_cache[player][r][c]
                distances.append(dist)
                score += dist * 3.0  # scale up distances for separation
                if (r, c) in board.home_camps[player]:
                    score += 6.0  # heavier home penalty to force exits

            if distances:
                distances.sort(reverse=True)
                # Re-emphasize up to three worst stragglers.
                score += sum(distances[:3])
            return score

        mine = side_score(self._root_player)
        theirs = side_score(self._opponent(self._root_player))
        return theirs - mine

    def _move_progress(self, board: HalmaBoard, player: str, move: Move) -> float:
        # Positive means the move gets closer to goal (with a tiny jump bonus).
        start_dist = board.distance_cache[player][move.start[0]][move.start[1]]
        dest_dist = board.distance_cache[player][move.dest[0]][move.dest[1]]
        jump_bonus = 0.25 if move.kind == "jump" else 0.0
        return (start_dist - dest_dist) + jump_bonus

    def _apply_move(self, board: HalmaBoard, player: str, move: Move) -> None:
        board.move_piece(player, move.start, move.dest)

    def _undo_move(self, board: HalmaBoard, player: str, move: Move) -> None:
        # Reverse of move_piece without validation overhead.
        board.board[move.dest[0]][move.dest[1]] = None
        board.board[move.start[0]][move.start[1]] = player
        board.pieces[player].remove(move.dest)
        board.pieces[player].add(move.start)
        if move.dest in board.home_camps[player]:
            board.home_counts[player] -= 1
        if move.start in board.home_camps[player]:
            board.home_counts[player] += 1

    def _opponent(self, player: str) -> str:
        return "R" if player == "G" else "G"

    def _record_prune(self, depth_remaining: int) -> None:
        self._prunes[depth_remaining] = self._prunes.get(depth_remaining, 0) + 1

    def _check_time(self) -> None:
        if time.monotonic() >= self._deadline:
            raise SearchTimeout()
