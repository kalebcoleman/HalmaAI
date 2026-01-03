"""
Board-level rules and mechanics for the Halma Part 1 project.

The :class:`HalmaBoard` class owns the abstract game state independent of any
user interface.  It tracks piece placement, legal move validation,
progress/blocking rules, scoring, and file/starting-board configuration.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


Coord = Tuple[int, int]


class HalmaBoard:
    """Encapsulates the rules, move validation, and scoring for Halma."""

    DIRECTIONS: Tuple[Coord, ...] = (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    )

    # Camp size controls how many starting pieces each side has.
    # For 8x8 we should have 10 pieces (triangle of side 4), matching standard Halma.
    CAMP_SIZES = {8: 4, 10: 4, 16: 5}
    PLAYER_ORDER = ("G", "R")

    def __init__(self, size: int) -> None:
        if size not in (8, 10, 16):
            raise ValueError("Board size must be 8, 10, or 16.")
        self.size = size
        self.board: List[List[Optional[str]]] = [
            [None for _ in range(size)] for _ in range(size)
        ]
        self.pieces: Dict[str, Set[Coord]] = {p: set() for p in self.PLAYER_ORDER}

        # Triangular camps sized by board; 8x8 gets 10 pieces (side 4).
        self.home_camps = {
            "G": self._compute_camp("top-left"),
            "R": self._compute_camp("bottom-right"),
        }
        self.goal_camps = {
            "G": set(self.home_camps["R"]),
            "R": set(self.home_camps["G"]),
        }

        self.goal_lists = {player: list(coords) for player, coords in self.goal_camps.items()}
        self.distance_cache = {
            player: self._precompute_distances(player) for player in self.PLAYER_ORDER
        }
        self.home_counts = {player: 0 for player in self.PLAYER_ORDER}

    # ------------------------------------------------------------------ Setup helpers

    def reset(self) -> None:
        for row in range(self.size):
            for col in range(self.size):
                self.board[row][col] = None
        for player in self.PLAYER_ORDER:
            self.pieces[player].clear()
            self.home_counts[player] = 0

    def setup_starting_positions(self) -> None:
        self.reset()
        for coord in self.home_camps["G"]:
            self._place_piece("G", coord)
        for coord in self.home_camps["R"]:
            self._place_piece("R", coord)

    def load_from_file(self, path: Path) -> None:
        if not path.exists():
            raise ValueError(f"Board file '{path}' does not exist.")

        rows: List[str] = []
        for raw in path.read_text().splitlines():
            # Remove trailing comments and whitespace to match the provided format.
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            rows.append("".join(line.split()))

        if len(rows) < self.size:
            raise ValueError(
                f"Board file must contain at least {self.size} rows (found {len(rows)})."
            )

        self.reset()

        for r in range(self.size):
            row_data = rows[r]
            if len(row_data) < self.size:
                raise ValueError(
                    f"Row {r + 1} in board file is too short (need {self.size} columns)."
                )
            for c in range(self.size):
                char = row_data[c].upper()
                if char in (".", "-", "_", "0"):
                    continue
                if char not in self.PLAYER_ORDER:
                    raise ValueError(
                        f"Invalid token '{row_data[c]}' at row {r + 1}, column {c + 1}."
                    )
                self._place_piece(char, (r, c))

    # ------------------------------------------------------------------ Move logic

    def legal_steps(self, coord: Coord, player: str) -> Iterable[Coord]:
        # Simple one-hop moves in any direction, filtered by progress rule.
        for dr, dc in self.DIRECTIONS:
            dest = (coord[0] + dr, coord[1] + dc)
            if not self._is_empty(dest):
                continue
            if self._progress_rule_allows(player, coord, dest):
                yield dest

    def legal_jumps(
        self,
        coord: Coord,
        player: str,
        origin: Coord,
        visited: Set[Coord],
    ) -> Set[Coord]:
        # DFS over chained jumps; origin/visited prevent looping back.
        reachable: Set[Coord] = set()

        def dfs(position: Coord, seen: Set[Coord]) -> None:
            for dr, dc in self.DIRECTIONS:
                mid = (position[0] + dr, position[1] + dc)
                dest = (position[0] + 2 * dr, position[1] + 2 * dc)
                if not self._in_bounds(mid) or not self._in_bounds(dest):
                    continue
                if self.board[mid[0]][mid[1]] is None:
                    continue
                if self.board[dest[0]][dest[1]] is not None:
                    continue
                if dest == origin or dest in seen:
                    continue
                if not self._progress_rule_allows(player, position, dest):
                    continue
                seen.add(dest)
                reachable.add(dest)
                dfs(dest, seen.copy())

        dfs(coord, set(visited))
        return reachable

    def move_piece(self, player: str, start: Coord, dest: Coord) -> None:
        if self.board[start[0]][start[1]] != player:
            raise ValueError("The starting square does not contain the player's piece.")
        if self.board[dest[0]][dest[1]] is not None:
            raise ValueError("Destination square is already occupied.")

        self.board[start[0]][start[1]] = None
        self.board[dest[0]][dest[1]] = player

        self.pieces[player].remove(start)
        self.pieces[player].add(dest)

        if start in self.home_camps[player]:
            self.home_counts[player] -= 1
        if dest in self.home_camps[player]:
            self.home_counts[player] += 1

    # ------------------------------------------------------------------ Query helpers

    def piece_at(self, coord: Coord) -> Optional[str]:
        if not self._in_bounds(coord):
            return None
        return self.board[coord[0]][coord[1]]

    def compute_scores(self) -> Dict[str, float]:
        # Lightweight “how far into camp” score for UI; not the AI utility.
        scores: Dict[str, float] = {}
        for player in self.PLAYER_ORDER:
            total = 0.0
            for coord in self.pieces[player]:
                if coord in self.goal_camps[player]:
                    total += 1.0
                else:
                    distance = self.distance_cache[player][coord[0]][coord[1]]
                    if distance > 0:
                        total += 1.0 / distance
            scores[player] = total
        return scores

    def has_won(self, player: str) -> bool:
        return all(square in self.pieces[player] for square in self.goal_camps[player])

    # ------------------------------------------------------------------ Internal helpers

    def _compute_camp(self, corner: str) -> Set[Coord]:
        camp_size = self.CAMP_SIZES.get(self.size, max(3, self.size // 3))
        coords: Set[Coord] = set()
        if corner == "top-left":
            for r in range(camp_size):
                for c in range(camp_size - r):
                    coords.add((r, c))
        elif corner == "bottom-right":
            for r in range(camp_size):
                row = self.size - 1 - r
                for c in range(camp_size - r):
                    col = self.size - 1 - c
                    coords.add((row, col))
        else:
            raise ValueError(f"Invalid camp corner '{corner}'.")
        return coords

    def _place_piece(self, player: str, coord: Coord) -> None:
        row, col = coord
        if self.board[row][col] is not None:
            raise ValueError(f"Square {coord} already contains a piece.")
        self.board[row][col] = player
        self.pieces[player].add(coord)
        if coord in self.home_camps[player]:
            self.home_counts[player] += 1

    def _is_empty(self, coord: Coord) -> bool:
        return self._in_bounds(coord) and self.board[coord[0]][coord[1]] is None

    def _in_bounds(self, coord: Coord) -> bool:
        row, col = coord
        return 0 <= row < self.size and 0 <= col < self.size

    def _precompute_distances(self, player: str) -> List[List[float]]:
        grid = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        targets = self.goal_lists[player]
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in self.goal_camps[player]:
                    grid[r][c] = 0.0
                else:
                    grid[r][c] = min(math.hypot(r - gr, c - gc) for (gr, gc) in targets)
        return grid

    def _progress_rule_allows(self, player: str, start: Coord, dest: Coord) -> bool:
        if not self._in_bounds(dest):
            return False
        if self.board[dest[0]][dest[1]] is not None:
            return False
        if start in self.goal_camps[player] and dest not in self.goal_camps[player]:
            return False

        start_dist = self.distance_cache[player][start[0]][start[1]]
        dest_dist = self.distance_cache[player][dest[0]][dest[1]]

        if dest_dist > start_dist + 1e-9:
            return False  # no backtracking; always make forward progress

        in_home = start in self.home_camps[player]
        if in_home and self.home_counts[player] > 0 and dest_dist >= start_dist - 1e-9:
            return False  # home pieces must move meaningfully forward

        return True
