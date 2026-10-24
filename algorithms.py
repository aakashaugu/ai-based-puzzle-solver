"""
algorithms.py - AI Search Algorithms for 8-Puzzle
=================================================
Academic Context:
- Uninformed Search: Breadth-First Search (BFS), Depth-First Search (DFS)
- Informed (Heuristic) Search: A* Search using Manhattan Distance
- Key Metrics Tracked:
    * Path Cost (Solution Length in Moves)
    * Time Complexity: Expanded Nodes, Generated Nodes, Execution Time (ms)
    * Space Complexity: Peak Frontier Size, Nodes in Memory
    * Optimality: Whether the returned solution is guaranteed shortest

Theoretical Background:
- BFS: Complete and optimal for unit step cost. Suffers from exponential memory O(b^d).
- DFS: Not optimal, not complete in infinite spaces. In finite graph with cycle checking,
  it explores deep paths first and often yields excessively long solutions.
- A* Search: Complete and optimal with an admissible and consistent heuristic (Manhattan).
  Prunes vast subtrees compared to BFS.

Authors / Mentors: AI Micro-Project Team
"""

import time
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

from puzzle import (
    GOAL_STATE,
    is_goal,
    is_valid_board,
    is_solvable,
    manhattan_distance,
    get_legal_moves,
    count_inversions,
)


@dataclass
class SearchResult:
    """Encapsulates the outcome, statistics, and solution path of an AI search run."""
    algorithm: str
    success: bool
    status: str  # 'SUCCESS', 'UNSOLVABLE', 'LIMIT_REACHED', 'NO_SOLUTION', 'INVALID_BOARD'
    path: List[Tuple[int, ...]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    moved_tiles: List[int] = field(default_factory=list)
    cost: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    nodes_in_memory: int = 0
    execution_time_ms: float = 0.0
    termination_reason: str = ""
    is_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for tabular comparisons and charts."""
        return {
            "Algorithm": self.algorithm,
            "Solved": "Yes" if self.success else "No",
            "Status": self.status,
            "Moves (Cost)": self.cost if self.success else None,
            "Nodes Expanded": self.nodes_expanded,
            "Nodes Generated": self.nodes_generated,
            "Peak Frontier": self.max_frontier_size,
            "Memory (Nodes)": self.nodes_in_memory,
            "Time (ms)": round(self.execution_time_ms, 2),
            "Optimal?": "Guaranteed" if self.is_optimal else "No",
            "Termination Reason": self.termination_reason
        }


def reconstruct_path(
    parent_map: Dict[Tuple[int, ...], Optional[Tuple[Tuple[int, ...], str, int]]],
    goal_state: Tuple[int, ...]
) -> Tuple[List[Tuple[int, ...]], List[str], List[int]]:
    """
    Reconstruct the sequence of board states, actions, and moved tiles
    by tracing back through parent pointers from the goal state.

    Args:
        parent_map: Dictionary mapping state -> (parent_state, action_taken, moved_tile)
        goal_state: The final state reached.

    Returns:
        Tuple: (path_of_states, actions_list, moved_tiles_list)
    """
    path = []
    actions = []
    moved_tiles = []

    curr = goal_state
    while curr is not None:
        path.append(curr)
        entry = parent_map.get(curr)
        if entry is None or entry[0] is None:
            break
        parent, action, tile = entry
        actions.append(action)
        moved_tiles.append(tile)
        curr = parent

    path.reverse()
    actions.reverse()
    moved_tiles.reverse()
    return path, actions, moved_tiles


# ==============================================================================
# A. BREADTH-FIRST SEARCH (BFS)
# ==============================================================================
def bfs_solve(
    initial_state: Tuple[int, ...],
    max_expansions: int = 50000,
    timeout_sec: float = 10.0
) -> SearchResult:
    """
    Breadth-First Search (BFS) graph-search implementation.

    Characteristics:
    - Uninformed Search Strategy.
    - Uses a FIFO queue (collections.deque).
    - Traverses the state space level-by-level (shallowest unexpanded node first).
    - Optimality: Guaranteed to find a shortest solution in terms of number of moves
      for unit step cost problems.
    - Memory: Stores all generated states in memory (O(b^d)), which can exhaust
      RAM for depths greater than ~12-15 on an 8-puzzle.

    Args:
        initial_state: 9-element starting board tuple.
        max_expansions: Safety limit on popped nodes to prevent browser freeze.
        timeout_sec: Maximum execution time in seconds.

    Returns:
        SearchResult with path, metrics, and execution diagnostics.
    """
    start_time = time.perf_counter()
    algo_name = "Breadth-First Search (BFS)"

    # Solvability and validation pre-checks
    if not is_valid_board(initial_state):
        return SearchResult(
            algorithm=algo_name,
            success=False,
            status="INVALID_BOARD",
            termination_reason="Board format is invalid. Must contain numbers 0-8 exactly once."
        )

    if not is_solvable(initial_state):
        inversions = count_inversions(initial_state)
        return SearchResult(
            algorithm=algo_name,
            success=False,
            status="UNSOLVABLE",
            termination_reason=f"State is mathematically unsolvable (Inversions = {inversions}, which is odd)."
        )

    # Edge case: already solved
    if is_goal(initial_state):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm=algo_name,
            success=True,
            status="SUCCESS",
            path=[initial_state],
            actions=[],
            moved_tiles=[],
            cost=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier_size=1,
            nodes_in_memory=1,
            execution_time_ms=elapsed_ms,
            termination_reason="Initial state is already the goal state (0 moves required).",
            is_optimal=True
        )

    # Queue stores states to expand: FIFO
    queue = deque([initial_state])
    # Parent map tracks: child_state -> (parent_state, action_taken, moved_tile)
    parent_map: Dict[Tuple[int, ...], Optional[Tuple[Tuple[int, ...], str, int]]] = {
        initial_state: None
    }

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while queue:
        # Time-out check
        if (time.perf_counter() - start_time) > timeout_sec:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm=algo_name,
                success=False,
                status="LIMIT_REACHED",
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(parent_map) + len(queue),
                execution_time_ms=elapsed_ms,
                termination_reason=f"BFS timed out after {timeout_sec:.1f}s. BFS memory and time grow exponentially with depth."
            )

        # Expansion limit check
        if nodes_expanded >= max_expansions:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm=algo_name,
                success=False,
                status="LIMIT_REACHED",
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(parent_map) + len(queue),
                execution_time_ms=elapsed_ms,
                termination_reason=f"BFS expansion limit ({max_expansions:,} nodes) reached. Demonstrates high memory/expansion overhead."
            )

        current = queue.popleft()
        nodes_expanded += 1

        # Goal test upon expansion
        if is_goal(current):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path, actions, moved_tiles = reconstruct_path(parent_map, current)
            return SearchResult(
                algorithm=algo_name,
                success=True,
                status="SUCCESS",
                path=path,
                actions=actions,
                moved_tiles=moved_tiles,
                cost=len(actions),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(parent_map) + len(queue),
                execution_time_ms=elapsed_ms,
                termination_reason="Optimal solution found by Breadth-First Search.",
                is_optimal=True
            )

        # Expand legal moves
        for action, tile, neighbor in get_legal_moves(current):
            if neighbor not in parent_map:
                parent_map[neighbor] = (current, action, tile)
                queue.append(neighbor)
                nodes_generated += 1

        if len(queue) > max_frontier_size:
            max_frontier_size = len(queue)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm=algo_name,
        success=False,
        status="NO_SOLUTION",
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        nodes_in_memory=len(parent_map),
        execution_time_ms=elapsed_ms,
        termination_reason="Queue exhausted without reaching goal state."
    )


# ==============================================================================
# B. DEPTH-FIRST SEARCH (DFS)
# ==============================================================================
def dfs_solve(
    initial_state: Tuple[int, ...],
    max_expansions: int = 25000,
    timeout_sec: float = 10.0
) -> SearchResult:
    """
    Depth-First Search (DFS) graph-search implementation with cycle prevention.

    Characteristics:
    - Uninformed Search Strategy.
    - Uses a LIFO stack (list).
    - Traverses the deepest branch first before backtracking.
    - Optimality: DOES NOT guarantee a shortest path. Can produce solutions with
      hundreds of moves for simple puzzles.
    - Safe Limit: Implements a strict node expansion ceiling so the process
      terminates safely rather than looping excessively.

    Args:
        initial_state: 9-element starting board tuple.
        max_expansions: Safety limit on popped nodes.
        timeout_sec: Maximum execution time in seconds.

    Returns:
        SearchResult with path, metrics, and execution diagnostics.
    """
    start_time = time.perf_counter()
    algo_name = "Depth-First Search (DFS)"

    # Solvability and validation pre-checks
    if not is_valid_board(initial_state):
        return SearchResult(
            algorithm=algo_name,
            success=False,
            status="INVALID_BOARD",
            termination_reason="Board format is invalid. Must contain numbers 0-8 exactly once."
        )

    if not is_solvable(initial_state):
        inversions = count_inversions(initial_state)
        return SearchResult(
            algorithm=algo_name,
            success=False,
            status="UNSOLVABLE",
            termination_reason=f"State is mathematically unsolvable (Inversions = {inversions}, which is odd)."
        )

    # Edge case: already solved
    if is_goal(initial_state):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm=algo_name,
            success=True,
            status="SUCCESS",
            path=[initial_state],
            actions=[],
            moved_tiles=[],
            cost=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier_size=1,
            nodes_in_memory=1,
            execution_time_ms=elapsed_ms,
            termination_reason="Initial state is already the goal state (0 moves required).",
            is_optimal=True
        )

    # Stack stores states to expand: LIFO
    stack = [initial_state]
    # Visited set to prevent infinite cycles in graph space
    visited = {initial_state}
    parent_map: Dict[Tuple[int, ...], Optional[Tuple[Tuple[int, ...], str, int]]] = {
        initial_state: None
    }

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while stack:
        # Time-out check
        if (time.perf_counter() - start_time) > timeout_sec:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm=algo_name,
                success=False,
                status="LIMIT_REACHED",
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(visited) + len(stack),
                execution_time_ms=elapsed_ms,
                termination_reason=f"DFS timed out after {timeout_sec:.1f}s. DFS explores deep non-optimal paths."
            )

        # Expansion limit check
        if nodes_expanded >= max_expansions:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm=algo_name,
                success=False,
                status="LIMIT_REACHED",
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(visited) + len(stack),
                execution_time_ms=elapsed_ms,
                termination_reason=(
                    f"DFS reached safety expansion limit ({max_expansions:,} nodes). "
                    "Notice: The puzzle IS solvable, but DFS plunged into a deep non-optimal branch."
                )
            )

        current = stack.pop()
        nodes_expanded += 1

        if is_goal(current):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path, actions, moved_tiles = reconstruct_path(parent_map, current)
            return SearchResult(
                algorithm=algo_name,
                success=True,
                status="SUCCESS",
                path=path,
                actions=actions,
                moved_tiles=moved_tiles,
                cost=len(actions),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(visited) + len(stack),
                execution_time_ms=elapsed_ms,
                termination_reason="Solution found by DFS (Note: DFS does NOT guarantee a shortest path).",
                is_optimal=(len(actions) == manhattan_distance(initial_state))  # Rare coincidence
            )

        # Successor generation (reverse list to expand natural order first)
        moves = get_legal_moves(current)
        for action, tile, neighbor in reversed(moves):
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = (current, action, tile)
                stack.append(neighbor)
                nodes_generated += 1

        if len(stack) > max_frontier_size:
            max_frontier_size = len(stack)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm=algo_name,
        success=False,
        status="NO_SOLUTION",
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        nodes_in_memory=len(visited),
        execution_time_ms=elapsed_ms,
        termination_reason="Stack exhausted without reaching goal state."
    )


# ==============================================================================
# C. A* SEARCH (INFORMED SEARCH WITH MANHATTAN DISTANCE)
# ==============================================================================
def astar_solve(
    initial_state: Tuple[int, ...],
    max_expansions: int = 100000,
    timeout_sec: float = 15.0
) -> SearchResult:
    """
    A* Search implementation using Manhattan Distance Heuristic.

    Evaluation Function:
        f(n) = g(n) + h(n)
        g(n) = cost incurred so far from initial state to node n (step cost = 1)
        h(n) = Manhattan Distance (sum of row and column tile distances to goal)

    Characteristics:
    - Informed (Heuristic) Search Strategy.
    - Priority Queue using Python `heapq`.
    - Tie-breaking: Uses (f, h, counter, state) to prioritize nodes closer to goal
      when f-scores are tied, significantly reducing runtime.
    - Optimality: Guaranteed to return the SHORTEST solution because Manhattan
      distance is both admissible and consistent.
    - Efficiency: Prunes massive branches compared to BFS, exploring orders of
      magnitude fewer nodes.

    Args:
        initial_state: 9-element starting board tuple.
        max_expansions: Upper bound on expansions.
        timeout_sec: Timeout threshold.

    Returns:
        SearchResult with optimal path, move sequence, and benchmark metrics.
    """
    start_time = time.perf_counter()
    algo_name = "A* Search (Manhattan)"

    # Solvability and validation pre-checks
    if not is_valid_board(initial_state):
        return SearchResult(
            algorithm=algo_name,
            success=False,
            status="INVALID_BOARD",
            termination_reason="Board format is invalid. Must contain numbers 0-8 exactly once."
        )

    if not is_solvable(initial_state):
        inversions = count_inversions(initial_state)
        return SearchResult(
            algorithm=algo_name,
            success=False,
            status="UNSOLVABLE",
            termination_reason=f"State is mathematically unsolvable (Inversions = {inversions}, which is odd)."
        )

    # Edge case: already solved
    if is_goal(initial_state):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm=algo_name,
            success=True,
            status="SUCCESS",
            path=[initial_state],
            actions=[],
            moved_tiles=[],
            cost=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier_size=1,
            nodes_in_memory=1,
            execution_time_ms=elapsed_ms,
            termination_reason="Initial state is already the goal state (0 moves required).",
            is_optimal=True
        )

    # Priority queue item: (f_score, h_score, tie_breaker_counter, state)
    counter = 0
    h_init = manhattan_distance(initial_state)
    f_init = 0 + h_init
    open_heap = [(f_init, h_init, counter, initial_state)]

    # g_score maps state -> lowest known cost g(n) from start
    g_score: Dict[Tuple[int, ...], int] = {initial_state: 0}

    # parent_map tracks: state -> (parent_state, action_taken, moved_tile)
    parent_map: Dict[Tuple[int, ...], Optional[Tuple[Tuple[int, ...], str, int]]] = {
        initial_state: None
    }

    # Closed set for expanded nodes (Graph Search to avoid re-expansion)
    closed_set = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while open_heap:
        # Time-out check
        if (time.perf_counter() - start_time) > timeout_sec:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm=algo_name,
                success=False,
                status="LIMIT_REACHED",
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(closed_set) + len(open_heap),
                execution_time_ms=elapsed_ms,
                termination_reason=f"A* timed out after {timeout_sec:.1f}s."
            )

        # Expansion limit check
        if nodes_expanded >= max_expansions:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm=algo_name,
                success=False,
                status="LIMIT_REACHED",
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(closed_set) + len(open_heap),
                execution_time_ms=elapsed_ms,
                termination_reason=f"A* search expansion limit ({max_expansions:,} nodes) reached."
            )

        f, h, _, current = heapq.heappop(open_heap)

        # Skip if already closed (since nodes with worse g might have been pushed earlier)
        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_expanded += 1

        # Goal reached
        if is_goal(current):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path, actions, moved_tiles = reconstruct_path(parent_map, current)
            return SearchResult(
                algorithm=algo_name,
                success=True,
                status="SUCCESS",
                path=path,
                actions=actions,
                moved_tiles=moved_tiles,
                cost=len(actions),
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                nodes_in_memory=len(closed_set) + len(open_heap),
                execution_time_ms=elapsed_ms,
                termination_reason="Optimal solution guaranteed and discovered by A* Search using Manhattan Heuristic.",
                is_optimal=True
            )

        curr_g = g_score[current]

        # Expand neighbors
        for action, tile, neighbor in get_legal_moves(current):
            if neighbor in closed_set:
                continue

            tentative_g = curr_g + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent_map[neighbor] = (current, action, tile)
                h_val = manhattan_distance(neighbor)
                f_val = tentative_g + h_val
                counter += 1
                heapq.heappush(open_heap, (f_val, h_val, counter, neighbor))
                nodes_generated += 1

        if len(open_heap) > max_frontier_size:
            max_frontier_size = len(open_heap)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm=algo_name,
        success=False,
        status="NO_SOLUTION",
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        nodes_in_memory=len(closed_set),
        execution_time_ms=elapsed_ms,
        termination_reason="Priority queue exhausted without finding goal."
    )


# ==============================================================================
# ALGORITHM DISPATCHER & COMPARISON SUITE
# ==============================================================================
ALGORITHM_FUNCTIONS = {
    "A* Search (Manhattan)": astar_solve,
    "Breadth-First Search (BFS)": bfs_solve,
    "Depth-First Search (DFS)": dfs_solve
}


def run_solver(
    state: Tuple[int, ...],
    algorithm_key: str,
    max_expansions: Optional[int] = None
) -> SearchResult:
    """Run the specified algorithm with sensible defaults."""
    func = ALGORITHM_FUNCTIONS.get(algorithm_key, astar_solve)
    if max_expansions:
        return func(state, max_expansions=max_expansions)
    return func(state)


def compare_all_algorithms(
    initial_state: Tuple[int, ...],
    bfs_limit: int = 40000,
    dfs_limit: int = 25000,
    astar_limit: int = 100000,
    timeout_sec: float = 8.0
) -> Dict[str, SearchResult]:
    """
    Run all three algorithms (BFS, DFS, A*) sequentially on the exact same
    initial puzzle board and collect empirical performance metrics.

    Args:
        initial_state: 9-element starting board.
        bfs_limit: Expansion ceiling for BFS to prevent browser/RAM crashes.
        dfs_limit: Expansion ceiling for DFS.
        astar_limit: Expansion ceiling for A*.
        timeout_sec: Per-algorithm timeout in seconds.

    Returns:
        Dictionary mapping algorithm name -> SearchResult.
    """
    results: Dict[str, SearchResult] = {}

    # Fast check: If puzzle is mathematically unsolvable, fail all quickly
    if not is_valid_board(initial_state):
        for name in ALGORITHM_FUNCTIONS.keys():
            results[name] = SearchResult(
                algorithm=name,
                success=False,
                status="INVALID_BOARD",
                termination_reason="Invalid board format."
            )
        return results

    if not is_solvable(initial_state):
        inv = count_inversions(initial_state)
        for name in ALGORITHM_FUNCTIONS.keys():
            results[name] = SearchResult(
                algorithm=name,
                success=False,
                status="UNSOLVABLE",
                termination_reason=f"Odd parity ({inv} inversions). Solvability theorem proves goal is unreachable."
            )
        return results

    # 1. Run A* Search
    results["A* Search (Manhattan)"] = astar_solve(
        initial_state, max_expansions=astar_limit, timeout_sec=timeout_sec
    )

    # 2. Run BFS
    results["Breadth-First Search (BFS)"] = bfs_solve(
        initial_state, max_expansions=bfs_limit, timeout_sec=timeout_sec
    )

    # 3. Run DFS
    results["Depth-First Search (DFS)"] = dfs_solve(
        initial_state, max_expansions=dfs_limit, timeout_sec=timeout_sec
    )

    return results
