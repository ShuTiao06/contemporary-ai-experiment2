"""Experiment 2: A* pathfinding, Minimax and Alpha-Beta search.

Run from this folder with:
    python experiment2.py
Only the Python standard library is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
import csv
import json
from pathlib import Path
from typing import Callable, Iterable

Coord = tuple[int, int]
Grid = list[list[object]]
DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0


def _entry_cost(cell: object, cost_mode: str) -> float | None:
    if cell == "X":
        return None
    if cost_mode == "unit":
        return 1.0
    if cell == "E":
        return 0.0
    if cell in {"S", "A", "B", "T1", "T2", "T3", "T4"}:
        return 1.0
    try:
        value = float(cell)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported map cell: {cell!r}") from exc
    if value < 0:
        raise ValueError("A* requires nonnegative movement costs")
    return value


def _heuristic(grid: Grid, node: Coord, goal: Coord, cost_mode: str, heuristic: str) -> float:
    if heuristic == "zero":
        return 0.0
    if heuristic != "manhattan":
        raise ValueError("heuristic must be 'zero' or 'manhattan'")
    distance = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    if cost_mode == "unit":
        return float(distance)
    traversable = [
        _entry_cost(cell, cost_mode)
        for row in grid for cell in row
        if cell not in {"X", "E"}
    ]
    min_step = min((c for c in traversable if c is not None), default=0.0)
    # The assignment makes entering E free, so only the steps before the final
    # move are charged. This is admissible when every other entry cost >= min_step.
    charged_steps = max(0, distance - 1) if grid[goal[0]][goal[1]] == "E" else distance
    return charged_steps * min_step


def astar(
    grid: Grid,
    start: Coord,
    goal: Coord,
    *,
    cost_mode: str = "unit",
    heuristic: str = "manhattan",
) -> tuple[list[Coord], int | float, SearchStats]:
    """Return a lowest-cost path (including both endpoints), cost and counters."""
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be a nonempty rectangle")
    rows, cols = len(grid), len(grid[0])
    for label, point in (("start", start), ("goal", goal)):
        if not (0 <= point[0] < rows and 0 <= point[1] < cols):
            raise ValueError(f"{label} is outside the grid: {point}")
        if grid[point[0]][point[1]] == "X":
            return [], -1, SearchStats()
    serial = count()
    g_score: dict[Coord, float] = {start: 0.0}
    parent: dict[Coord, Coord] = {}
    frontier: list[tuple[float, float, int, Coord]] = [
        (_heuristic(grid, start, goal, cost_mode, heuristic), 0.0, next(serial), start)
    ]
    stats = SearchStats()
    while frontier:
        _, g, _, current = heappop(frontier)
        if g != g_score.get(current):
            continue
        stats.expanded += 1
        if current == goal:
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            return path, int(g) if g.is_integer() else g, stats
        for dr, dc in DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)
            nr, nc = neighbor
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            step = _entry_cost(grid[nr][nc], cost_mode)
            if step is None:
                continue
            tentative = g + step
            if tentative < g_score.get(neighbor, float("inf")):
                parent[neighbor] = current
                g_score[neighbor] = tentative
                h = _heuristic(grid, neighbor, goal, cost_mode, heuristic)
                heappush(frontier, (tentative + h, tentative, next(serial), neighbor))
                stats.generated += 1
    return [], -1, stats


def maze_shortest_path(start: Coord, goal: Coord, grid: Grid) -> list[Coord]:
    """Assignment task 1 API: return a shortest four-neighbor path or []."""
    path, _, _ = astar(grid, start, goal, cost_mode="unit", heuristic="manhattan")
    return path


def weighted_path_cost(start: Coord, goal: Coord, grid: Grid) -> int:
    """Assignment task 2 API: return minimum entry-cost sum or -1."""
    _, cost, _ = astar(grid, start, goal, cost_mode="weighted", heuristic="manhattan")
    return int(cost)


@dataclass
class TreeNode:
    value: int | None = None
    children: list["TreeNode"] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []
        if self.value is None and not self.children:
            raise ValueError("a node must have a value or children")
        if self.value is not None and self.children:
            raise ValueError("a leaf cannot also have children")


@dataclass
class GameStats:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0


@dataclass(frozen=True)
class GameState:
    pos_a: str
    pos_b: str
    remaining: tuple[str, ...]
    score_a: int
    score_b: int
    player: str


TREASURE_GRID: Grid = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
TREASURE_POS: dict[str, Coord] = {
    "A": (0, 0), "B": (6, 6), "T1": (0, 6),
    "T2": (2, 2), "T3": (4, 4), "T4": (6, 0),
}
TREASURE_VALUE = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


def all_pairwise_distances() -> tuple[list[str], list[list[int]]]:
    labels = ["A", "B", "T1", "T2", "T3", "T4"]
    matrix: list[list[int]] = []
    for source in labels:
        row = []
        for target in labels:
            _, cost, _ = astar(
                TREASURE_GRID, TREASURE_POS[source], TREASURE_POS[target],
                cost_mode="weighted", heuristic="manhattan",
            )
            row.append(int(cost))
        matrix.append(row)
    return labels, matrix


def _game_child(state: GameState, target: str, distances: dict[tuple[str, str], int]) -> GameState:
    if state.player == "A":
        gain = TREASURE_VALUE[target] - distances[(state.pos_a, target)]
        return GameState(target, state.pos_b, tuple(t for t in state.remaining if t != target),
                         state.score_a + gain, state.score_b, "B")
    gain = TREASURE_VALUE[target] - distances[(state.pos_b, target)]
    return GameState(state.pos_a, target, tuple(t for t in state.remaining if t != target),
                     state.score_a, state.score_b + gain, "A")


def _ordered_targets(state: GameState, distances: dict[tuple[str, str], int], order: str) -> list[str]:
    targets = list(state.remaining)
    if order == "natural":
        return targets
    if order != "greedy":
        raise ValueError("order must be 'natural' or 'greedy'")
    pos = state.pos_a if state.player == "A" else state.pos_b
    # For either player, an earlier move with a larger immediate gain is useful:
    # MAX sees a larger positive utility first; MIN sees a larger subtraction first.
    return sorted(targets, key=lambda t: (-(TREASURE_VALUE[t] - distances[(pos, t)]), t))


def minimax_game(depth: int, distances: dict[tuple[str, str], int], order: str = "natural") -> tuple[str | None, int, GameStats]:
    stats = GameStats()
    stats.nodes = 1  # Include the root in the search-node count.
    start = GameState("A", "B", tuple(TREASURE_VALUE), 0, 0, "A")

    def visit(state: GameState, left: int) -> int:
        stats.nodes += 1
        if left == 0 or not state.remaining:
            stats.leaves += 1
            return state.score_a - state.score_b
        values = [visit(_game_child(state, t, distances), left - 1)
                  for t in _ordered_targets(state, distances, order)]
        return max(values) if state.player == "A" else min(values)

    best_target = None
    best_value = -float("inf")
    for target in _ordered_targets(start, distances, order):
        value = visit(_game_child(start, target, distances), depth - 1)
        if value > best_value:
            best_target, best_value = target, value
    return best_target, int(best_value), stats


def alphabeta_game(depth: int, distances: dict[tuple[str, str], int], order: str = "natural") -> tuple[str | None, int, GameStats]:
    stats = GameStats()
    stats.nodes = 1  # Include the root in the search-node count.
    start = GameState("A", "B", tuple(TREASURE_VALUE), 0, 0, "A")

    def visit(state: GameState, left: int, alpha: float, beta: float) -> int:
        stats.nodes += 1
        if left == 0 or not state.remaining:
            stats.leaves += 1
            return state.score_a - state.score_b
        if state.player == "A":
            value = -float("inf")
            for target in _ordered_targets(state, distances, order):
                value = max(value, visit(_game_child(state, target, distances), left - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return int(value)
        value = float("inf")
        for target in _ordered_targets(state, distances, order):
            value = min(value, visit(_game_child(state, target, distances), left - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return int(value)

    best_target = None
    best_value = -float("inf")
    alpha, beta = -float("inf"), float("inf")
    for target in _ordered_targets(start, distances, order):
        value = visit(_game_child(start, target, distances), depth - 1, alpha, beta)
        if value > best_value:
            best_target, best_value = target, value
        alpha = max(alpha, best_value)
    return best_target, int(best_value), stats


def alpha_beta_tree(root: TreeNode, *, reverse: bool = False) -> tuple[int, GameStats]:
    stats = GameStats()

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float) -> int:
        stats.nodes += 1
        if node.value is not None:
            stats.leaves += 1
            return node.value
        children = list(node.children or [])
        if reverse:
            children.reverse()
        if maximizing:
            value = -float("inf")
            for child in children:
                value = max(value, visit(child, False, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
            return int(value)
        value = float("inf")
        for child in children:
            value = min(value, visit(child, True, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return int(value)

    return visit(root, True, -float("inf"), float("inf")), stats


def solve_minimax(root: TreeNode) -> int:
    """Assignment task 3 API: minimax value for a MAX root."""
    return minimax_tree(root)[0]


def solve_alpha_beta(root: TreeNode) -> int:
    """Assignment task 3 API: alpha-beta value for a MAX root."""
    return alpha_beta_tree(root)[0]


def minimax_tree(root: TreeNode, *, reverse: bool = False) -> tuple[int, GameStats]:
    stats = GameStats()

    def visit(node: TreeNode, maximizing: bool) -> int:
        stats.nodes += 1
        if node.value is not None:
            stats.leaves += 1
            return node.value
        children = list(node.children or [])
        if reverse:
            children.reverse()
        values = [visit(child, not maximizing) for child in children]
        return max(values) if maximizing else min(values)

    return visit(root, True), stats


def _sample_tree() -> TreeNode:
    max1 = TreeNode(children=[TreeNode(value=3), TreeNode(value=5)])
    max2_left = TreeNode(children=[TreeNode(value=6), TreeNode(value=9)])
    max2_right = TreeNode(children=[TreeNode(value=6), TreeNode(value=9)])
    max3 = TreeNode(children=[TreeNode(value=1), TreeNode(value=2)])
    min1 = TreeNode(children=[max1, max2_left])
    min2 = TreeNode(children=[max2_right, max3])
    return TreeNode(children=[min1, min2])


def run_experiments(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    maze = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    maze_records = []
    for h in ("zero", "manhattan"):
        path, cost, stats = astar(maze, (0, 0), (3, 4), cost_mode="unit", heuristic=h)
        maze_records.append({"heuristic": h, "cost": cost, "expanded": stats.expanded,
                             "generated": stats.generated, "path": path})

    weighted = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    weighted_records = []
    for h in ("zero", "manhattan"):
        path, cost, stats = astar(weighted, (0, 0), (4, 4), cost_mode="weighted", heuristic=h)
        weighted_records.append({"heuristic": h, "cost": cost, "expanded": stats.expanded,
                                 "generated": stats.generated, "path": path})

    labels, matrix = all_pairwise_distances()
    distances = {(a, b): matrix[i][j] for i, a in enumerate(labels) for j, b in enumerate(labels)}
    tree_results = {}
    for reverse in (False, True):
        root = _sample_tree()
        mm_value, mm_stats = minimax_tree(root, reverse=reverse)
        ab_value, ab_stats = alpha_beta_tree(root, reverse=reverse)
        tree_results["reverse" if reverse else "left_to_right"] = {
            "minimax": {"value": mm_value, "nodes": mm_stats.nodes, "leaves": mm_stats.leaves},
            "alpha_beta": {"value": ab_value, "nodes": ab_stats.nodes,
                            "leaves": ab_stats.leaves, "cutoffs": ab_stats.cutoffs},
        }

    game_results = {}
    for depth in (2, 4):
        game_results[str(depth)] = {}
        for algo, solver in (("minimax", minimax_game), ("alpha_beta_natural", alphabeta_game),
                             ("alpha_beta_greedy", alphabeta_game)):
            order = "greedy" if algo.endswith("greedy") else "natural"
            target, value, stats = solver(depth, distances, order=order)
            game_results[str(depth)][algo] = {
                "best_target": target, "best_value": value, "nodes": stats.nodes,
                "leaves": stats.leaves, "cutoffs": stats.cutoffs,
            }

    results: dict[str, object] = {
        "maze": maze_records, "weighted": weighted_records,
        "treasure_labels": labels, "distance_matrix": matrix,
        "tree": tree_results, "game": game_results,
    }
    (output_dir / "experiment2_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "distance_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["from/to", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    run_experiments(Path(__file__).resolve().parent)
