"""
Tkinter-based interface layer for Halma with AI support.

Layout includes a right-hand sidebar showing scores, goals-to-win, whose turn,
an AI "thinking" indicator, and search stats (boards/time/prunes).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import tkinter as tk

# Package/script import shim
try:  # pragma: no cover
    from .ai import HalmaAI, Move, SearchStats  # type: ignore
    from .board import Coord, HalmaBoard  # type: ignore
except ImportError:  # pragma: no cover
    from halma.ai import HalmaAI, Move, SearchStats  # type: ignore
    from halma.board import Coord, HalmaBoard  # type: ignore


@dataclass(frozen=True)
class MoveDescriptor:
    kind: str  # "step" or "jump"


class HalmaGUI:
    PLAYER_NAMES = {"G": "GREEN", "R": "RED"}
    PLAYER_FILL = {"G": "#2ecc71", "R": "#d64550"}
    PLAYER_OUTLINE = {"G": "#1b7d46", "R": "#812027"}
    LAST_MOVE_COLORS = {"G": "#c9f3d6", "R": "#ffd6cc"}

    STEP_HILITE = "#7CC0FF"
    JUMP_HILITE = "#85E085"
    GRID_BASE = "#f3e8d8"
    GRID_ALT = "#e9dec9"
    GRID_OUTLINE = "#cbb69c"

    def __init__(
        self,
        root: tk.Tk,
        board: HalmaBoard,
        *,
        time_limit: float,
        human_color: str,
        ai_color: Optional[str] = None,
        search_depth: int = 3,
        ai_time_limit: Optional[float] = None,
        use_alpha_beta: bool = True,
    ) -> None:
        # Build the UI: board on the left, status/metrics on the right.
        self.root = root
        self.board = board
        self.time_limit = time_limit
        self.human_color = human_color.upper()
        self.ai_time_limit = ai_time_limit or time_limit
        self.use_alpha_beta = use_alpha_beta
        self.search_depth = search_depth

        self.cell_size = 70 if board.size <= 10 else 48
        self.margin = 40
        canvas_span = self.margin * 2 + board.size * self.cell_size

        # Containers
        self.main_frame = tk.Frame(root, bg="#e6e6e6")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.main_frame,
            width=canvas_span,
            height=canvas_span,
            bg=self.GRID_BASE,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, padx=(10, 0), pady=10, sticky="nw")

        self.sidebar = tk.Frame(self.main_frame, width=220, bg="#f5f5f5")
        self.sidebar.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
        self.sidebar.grid_propagate(False)

        self.turn_var = tk.StringVar()
        self.timer_var = tk.StringVar()
        self.score_var = tk.StringVar()
        self.goal_red_var = tk.StringVar()
        self.goal_green_var = tk.StringVar()
        self.thinking_var = tk.StringVar()
        self.stats_var = tk.StringVar()
        self.status_var = tk.StringVar()

        tk.Label(self.sidebar, textvariable=self.turn_var, font=("Helvetica", 13, "bold"), bg="#f5f5f5").pack(
            anchor="w", pady=(6, 4)
        )
        tk.Label(self.sidebar, textvariable=self.timer_var, font=("Helvetica", 12), bg="#f5f5f5").pack(
            anchor="w", pady=(0, 4)
        )
        tk.Label(self.sidebar, textvariable=self.score_var, font=("Helvetica", 12), bg="#f5f5f5").pack(
            anchor="w", pady=(0, 8)
        )
        tk.Label(
            self.sidebar, textvariable=self.goal_red_var, font=("Helvetica", 11), fg="#d64550", bg="#f5f5f5"
        ).pack(anchor="w", pady=(0, 2))
        tk.Label(
            self.sidebar, textvariable=self.goal_green_var, font=("Helvetica", 11), fg="#1b7d46", bg="#f5f5f5"
        ).pack(anchor="w", pady=(0, 8))
        tk.Label(
            self.sidebar, textvariable=self.thinking_var, font=("Helvetica", 11, "italic"), fg="#1d4ed8", bg="#f5f5f5"
        ).pack(anchor="w", pady=(4, 6))
        tk.Label(self.sidebar, textvariable=self.stats_var, font=("Helvetica", 10), fg="#444", bg="#f5f5f5").pack(
            anchor="w", pady=(0, 4)
        )
        tk.Label(self.sidebar, textvariable=self.status_var, font=("Helvetica", 10), fg="#444", bg="#f5f5f5", wraplength=180).pack(
            anchor="w", pady=(4, 4)
        )

        self.controls = tk.Frame(self.main_frame, bg="#e6e6e6")
        self.controls.grid(row=1, column=0, pady=(0, 10), padx=(10, 0), sticky="w")
        self.finish_button = tk.Button(
            self.controls, text="End Jump Chain", command=self._finish_jump_chain, state=tk.DISABLED
        )
        self.finish_button.pack(side=tk.LEFT, padx=4, pady=4)

        self.canvas.bind("<Button-1>", self._handle_click)

        self.piece_items: Dict[Coord, int] = {}
        self.selected_piece: Optional[Coord] = None
        self.selection_anchor: Optional[Coord] = None
        self.jump_origin: Optional[Coord] = None
        self.jump_history: Set[Coord] = set()
        self.active_jump = False
        self.jump_made = False
        self.move_targets: Dict[Coord, MoveDescriptor] = {}

        self.selection_outline: Optional[int] = None
        self.move_highlights: List[int] = []
        self.last_move_marks: Dict[str, List[int]] = {player: [] for player in HalmaBoard.PLAYER_ORDER}

        self.current_player = "G"
        self.move_count = 0
        self.scores = self.board.compute_scores()
        self.ai_move_in_progress = False
        self.ai_agents = self._build_ai_agents(ai_color)

        self.timer_job: Optional[str] = None
        self.turn_start_time = time.monotonic()
        self.game_over = False

        self._draw_grid()
        self._draw_labels()
        self._render_all_pieces()

        self._update_scores()
        self._update_status("Game start.")
        self._start_turn_timer()
        self._trigger_ai_move_if_needed()

    # ---------------- Drawing

    def _draw_grid(self) -> None:
        for r in range(self.board.size):
            for c in range(self.board.size):
                x0, y0, x1, y1 = self._cell_bounds((r, c))
                color = self.GRID_BASE if (r + c) % 2 == 0 else self.GRID_ALT
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=self.GRID_OUTLINE)

        # Shade camps for clarity
        for coord in self.board.home_camps["G"]:
            self._draw_camp_cell(coord, "#c8eedd")
        for coord in self.board.home_camps["R"]:
            self._draw_camp_cell(coord, "#f7d4d4")

    def _draw_camp_cell(self, coord: Coord, color: str) -> None:
        x0, y0, x1, y1 = self._cell_bounds(coord)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=self.GRID_OUTLINE)

    def _draw_labels(self) -> None:
        letters = [chr(ord("a") + idx) for idx in range(self.board.size)]
        for c, letter in enumerate(letters):
            x = self.margin + c * self.cell_size + self.cell_size / 2
            self.canvas.create_text(x, self.margin - 18, text=letter, font=("Helvetica", 12, "bold"))
            self.canvas.create_text(
                x,
                self.margin + self.board.size * self.cell_size + 18,
                text=letter,
                font=("Helvetica", 12, "bold"),
            )
        for r in range(self.board.size):
            y = self.margin + r * self.cell_size + self.cell_size / 2
            label = str(r + 1)
            self.canvas.create_text(self.margin - 18, y, text=label, font=("Helvetica", 12, "bold"))
            self.canvas.create_text(
                self.margin + self.board.size * self.cell_size + 18,
                y,
                text=label,
                font=("Helvetica", 12, "bold"),
            )

    def _render_all_pieces(self) -> None:
        for player, coords in self.board.pieces.items():
            for coord in coords:
                self.piece_items[coord] = self._draw_piece(player, coord)

    def _draw_piece(self, player: str, coord: Coord) -> int:
        x0, y0, x1, y1 = self._cell_bounds(coord, inset=8)
        return self.canvas.create_oval(
            x0,
            y0,
            x1,
            y1,
            fill=self.PLAYER_FILL[player],
            outline=self.PLAYER_OUTLINE[player],
            width=2,
        )

    def _cell_bounds(self, coord: Coord, inset: int = 0):
        row, col = coord
        x0 = self.margin + col * self.cell_size + inset
        y0 = self.margin + row * self.cell_size + inset
        x1 = x0 + self.cell_size - 2 * inset
        y1 = y0 + self.cell_size - 2 * inset
        return x0, y0, x1, y1

    def _highlight_piece(self, coord: Coord) -> None:
        self._clear_selection_outline()
        x0, y0, x1, y1 = self._cell_bounds(coord, inset=4)
        self.selection_outline = self.canvas.create_oval(
            x0, y0, x1, y1, outline="#f0b400", width=4
        )

    def _clear_selection_outline(self) -> None:
        if self.selection_outline is not None:
            self.canvas.delete(self.selection_outline)
            self.selection_outline = None

    def _draw_move_highlight(self, coord: Coord, color: str) -> None:
        x0, y0, x1, y1 = self._cell_bounds(coord, inset=6)
        rect = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=color, fill=color, stipple="gray25", width=3
        )
        self.move_highlights.append(rect)

    def _clear_move_highlights(self) -> None:
        for rect in self.move_highlights:
            self.canvas.delete(rect)
        self.move_highlights.clear()
        self.move_targets.clear()

    def _record_last_move(self, player: str, start: Coord, end: Coord) -> None:
        for item in self.last_move_marks[player]:
            self.canvas.delete(item)
        self.last_move_marks[player] = []
        for coord in (start, end):
            x0, y0, x1, y1 = self._cell_bounds(coord)
            rect = self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=self.LAST_MOVE_COLORS[player],
                width=4,
            )
            self.last_move_marks[player].append(rect)

    # ---------------- Interaction

    def _coord_from_click(self, x: float, y: float) -> Optional[Coord]:
        board_x = x - self.margin
        board_y = y - self.margin
        if board_x < 0 or board_y < 0:
            return None
        col = int(board_x // self.cell_size)
        row = int(board_y // self.cell_size)
        coord = (row, col)
        return coord if 0 <= row < self.board.size and 0 <= col < self.board.size else None

    def _handle_click(self, event: tk.Event) -> None:
        if self.game_over:
            return
        if self.ai_move_in_progress or self.current_player in self.ai_agents:
            self._update_status("AI turn - please wait.")
            return

        coord = self._coord_from_click(event.x, event.y)
        if coord is None:
            return

        if coord in self.move_targets:
            self._execute_move(coord)
            return

        occupant = self.board.piece_at(coord)
        if occupant != self.current_player:
            self._update_status("Select one of your own pieces.")
            return

        if self.active_jump:
            if coord == self.selected_piece:
                self._finish_jump_chain()
            else:
                self._update_status("Finish the current jump chain first.")
            return

        if self.selected_piece == coord and not self.active_jump:
            self._clear_selection()
            self._update_status("Selection cleared.")
            return

        self._select_piece(coord)

    def _select_piece(self, coord: Coord) -> None:
        # Reset jump-chain tracking and show fresh highlights.
        self.selected_piece = coord
        self.selection_anchor = coord
        self.jump_origin = coord
        self.jump_history = {coord}
        self.active_jump = False
        self.jump_made = False
        self.finish_button.config(state=tk.DISABLED)
        self._highlight_piece(coord)
        self._show_available_moves(coord, include_steps=True)
        if not self.move_targets:
            self._update_status("No legal moves for that piece.")
        else:
            self._update_status("Piece selected - choose a highlighted destination.")

    def _clear_selection(self) -> None:
        self.selected_piece = None
        self.selection_anchor = None
        self.jump_origin = None
        self.jump_history.clear()
        self.active_jump = False
        self.jump_made = False
        self.finish_button.config(state=tk.DISABLED)
        self._clear_selection_outline()
        self._clear_move_highlights()

    def _show_available_moves(self, coord: Coord, *, include_steps: bool) -> None:
        self._clear_move_highlights()
        player = self.current_player
        if include_steps:
            for dest in self.board.legal_steps(coord, player):
                self.move_targets[dest] = MoveDescriptor("step")
                self._draw_move_highlight(dest, self.STEP_HILITE)

        origin = self.jump_origin or coord
        visited = set(self.jump_history) if self.active_jump else {coord}
        for dest in self.board.legal_jumps(coord, player, origin, visited):
            self.move_targets[dest] = MoveDescriptor("jump")
            self._draw_move_highlight(dest, self.JUMP_HILITE)

    def _execute_move(self, dest: Coord) -> None:
        move = self.move_targets[dest]
        assert self.selected_piece is not None
        start = self.selected_piece
        player = self.current_player

        self.board.move_piece(player, start, dest)
        self._update_piece_coords(start, dest)
        self.selected_piece = dest

        if move.kind == "jump":
            if not self.active_jump:
                self.jump_origin = start
            self.active_jump = True
            self.jump_made = True
            self.jump_history.add(dest)
            self._highlight_piece(dest)
            self._show_available_moves(dest, include_steps=False)
            if not self.move_targets:
                self._finish_move(dest)
            else:
                self.finish_button.config(state=tk.NORMAL)
                self._update_status("Jump chain active - continue or click Finish.")
        else:
            self._finish_move(dest)

    def _update_piece_coords(self, start: Coord, dest: Coord) -> None:
        item_id = self.piece_items.pop(start)
        x0, y0, x1, y1 = self._cell_bounds(dest, inset=8)
        self.canvas.coords(item_id, x0, y0, x1, y1)
        self.piece_items[dest] = item_id

    def _finish_jump_chain(self) -> None:
        if self.active_jump and self.jump_made:
            self._finish_move(self.selected_piece)
        else:
            self._update_status("No active jump chain to finish.")

    def _finish_move(self, final_coord: Optional[Coord]) -> None:
        if final_coord is None:
            return
        start_coord = self.selection_anchor or final_coord
        self._record_last_move(self.current_player, start_coord, final_coord)
        self._clear_selection()
        self._advance_turn()

    def _advance_turn(self) -> None:
        self.move_count += 1
        self._update_scores()
        self._cancel_timer()
        self.thinking_var.set("")

        if self.board.has_won(self.current_player):
            self._declare_winner(self.current_player, "goal achieved")
            return

        self.current_player = "R" if self.current_player == "G" else "G"
        self._update_status(f"{self.PLAYER_NAMES[self.current_player]}'s turn.")
        self._start_turn_timer()
        self._trigger_ai_move_if_needed()

    # ---------------- Status / scoring

    def _update_scores(self) -> None:
        self.scores = self.board.compute_scores()
        self.score_var.set(
            f"Scores - Green: {self.scores['G']:.2f} | Red: {self.scores['R']:.2f}"
        )
        self._update_goals_remaining()
        self._update_turn_banner()

    def _update_goals_remaining(self) -> None:
        green_remaining = len([c for c in self.board.goal_camps["G"] if c not in self.board.pieces["G"]])
        red_remaining = len([c for c in self.board.goal_camps["R"] if c not in self.board.pieces["R"]])
        self.goal_red_var.set(f"Goals until Red wins: {red_remaining}")
        self.goal_green_var.set(f"Goals until Green wins: {green_remaining}")

    def _update_turn_banner(self) -> None:
        who = self.PLAYER_NAMES[self.current_player]
        if self.current_player == "G":
            self.turn_var.set(f"{who}'s turn")
        else:
            self.turn_var.set(f"{who}'s turn")

    def _declare_winner(self, winner: str, reason: str) -> None:
        self.game_over = True
        self._cancel_timer()
        self._update_status(f"{self.PLAYER_NAMES[winner]} wins by {reason}!")
        cycles = self.move_count // 2
        summary = (
            f"{self.PLAYER_NAMES[winner]} wins ({reason}).\n\n"
            f"Move cycles: {cycles}\n"
            f"Final scores - Green {self.scores['G']:.2f}, Red {self.scores['R']:.2f}."
        )
        self._show_game_over_dialog(summary)
        self.finish_button.config(state=tk.DISABLED)

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    # ---------------- Timer

    def _start_turn_timer(self) -> None:
        # Basic per-turn countdown; keeps pressure on both sides.
        self.turn_start_time = time.monotonic()
        self.timer_var.set(f"Timer: {self.time_limit:04.1f}s")
        self._schedule_timer_tick()

    def _schedule_timer_tick(self) -> None:
        elapsed = time.monotonic() - self.turn_start_time
        remaining = self.time_limit - elapsed
        if remaining <= 0:
            self.timer_var.set("Timer: 0.0s")
            self._handle_forfeit(self.current_player)
            return
        self.timer_var.set(f"Timer: {remaining:04.1f}s")
        self.timer_job = self.root.after(100, self._schedule_timer_tick)

    def _cancel_timer(self) -> None:
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

    def _handle_forfeit(self, loser: str) -> None:
        if self.game_over:
            return
        winner = "R" if loser == "G" else "G"
        self._declare_winner(winner, "time forfeit")

    # ---------------- Game over dialog

    def _show_game_over_dialog(self, summary: str) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Game Over")
        dialog.transient(self.root)
        dialog.resizable(False, False)

        tk.Label(dialog, text=summary, justify="left", padx=12, pady=10, font=("Helvetica", 11)).pack()

        button_row = tk.Frame(dialog, padx=12, pady=8)
        button_row.pack(fill=tk.X)

        tk.Button(button_row, text="Replay", width=10, command=lambda: self._restart_game(dialog)).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        tk.Button(button_row, text="Exit", width=10, command=self.root.destroy).pack(side=tk.LEFT)

        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift(self.root)
        dialog.geometry(f"+{self.root.winfo_rootx() + 50}+{self.root.winfo_rooty() + 50}")
        dialog.focus_set()
        try:
            dialog.grab_set()
        except tk.TclError:
            pass

    def _restart_game(self, dialog: tk.Toplevel) -> None:
        dialog.destroy()
        self._cancel_timer()
        self.game_over = False
        self.move_count = 0
        self.current_player = "G"
        self.board.setup_starting_positions()
        self.piece_items.clear()
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_labels()
        self._render_all_pieces()
        self._update_scores()
        self._start_turn_timer()
        self._trigger_ai_move_if_needed()

    # ---------------- AI

    def _build_ai_agents(self, ai_color: Optional[str]) -> Dict[str, HalmaAI]:
        mapping: Dict[str, HalmaAI] = {}
        if ai_color is None or ai_color.lower() == "none":
            return mapping
        ai_color = ai_color.lower()
        if ai_color in ("green", "both"):
            mapping["G"] = HalmaAI(max_depth=self.search_depth, use_alpha_beta=self.use_alpha_beta)
        if ai_color in ("red", "both"):
            mapping["R"] = HalmaAI(max_depth=self.search_depth, use_alpha_beta=self.use_alpha_beta)
        return mapping

    def _trigger_ai_move_if_needed(self) -> None:
        if self.game_over:
            return
        ai = self.ai_agents.get(self.current_player)
        if ai is None or self.ai_move_in_progress:
            return
        self.ai_move_in_progress = True
        self.thinking_var.set("AI is thinking...")
        # Think in a background thread so the UI doesn't freeze on us.
        thread = threading.Thread(target=self._run_ai_turn, args=(ai, self.current_player), daemon=True)
        thread.start()

    def _run_ai_turn(self, ai: HalmaAI, player: str) -> None:
        move, stats = ai.choose_move(
            self.board,
            player,
            time_limit=self.ai_time_limit,
            max_depth=self.search_depth,
            use_alpha_beta=self.use_alpha_beta,
        )
        self.root.after(0, lambda: self._finalize_ai_move(player, move, stats))

    def _finalize_ai_move(self, player: str, move: Optional[Move], stats: SearchStats) -> None:
        self.ai_move_in_progress = False
        self.thinking_var.set("")
        if self.game_over or self.current_player != player:
            return
        if move is None:
            self._handle_forfeit(player)
            return

        self.board.move_piece(player, move.start, move.dest)
        self._update_piece_coords(move.start, move.dest)
        self._record_last_move(player, move.start, move.dest)
        self._clear_selection()

        summary = (
            f"AI moved {self._coord_label(move.start)} -> {self._coord_label(move.dest)} "
            f"(val={stats.best_value:.2f}, depth={stats.depth_reached}, "
            f"nodes={stats.nodes_evaluated}, time={stats.elapsed:.2f}s)"
        )
        # Surface stats in GUI + console; makes demos and grading a lot easier.
        boards_line = f"Analyzed {stats.nodes_evaluated} boards in {stats.elapsed:.4f} seconds."
        self.stats_var.set(boards_line)
        print(boards_line)
        print(f"AI: Best move has utility {stats.best_value:.2f}")
        for line in self._format_prune_lines(stats):
            print(line)
        self._update_scores()
        self._update_status(summary)
        self._advance_turn()

    def _coord_label(self, coord: Coord) -> str:
        row, col = coord
        return f"{chr(ord('a') + col)}{row + 1}"

    def _format_prune_stats(self, stats: SearchStats) -> str:
        if stats.depth_reached == 0:
            return "Prunes: none"
        parts: List[str] = []
        for depth_idx in range(stats.depth_reached):
            depth_remaining = stats.depth_reached - depth_idx
            count = stats.prunes_by_depth.get(depth_remaining, 0)
            parts.append(f"{count} prunes at depth {depth_idx}")
        return " | ".join(parts)

    def _format_prune_lines(self, stats: SearchStats) -> List[str]:
        if stats.depth_reached == 0:
            return ["0 prunes at depth 0"]
        lines: List[str] = []
        for depth_idx in range(stats.depth_reached):
            depth_remaining = stats.depth_reached - depth_idx
            count = stats.prunes_by_depth.get(depth_remaining, 0)
            lines.append(f"{count} prunes at depth {depth_idx}")
        return lines
