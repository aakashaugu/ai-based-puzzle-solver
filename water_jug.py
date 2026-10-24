"""
water_jug.py - Water Jug Problem Space and Search Algorithms
============================================================
Academic Context:
- Artificial Intelligence Syllabus: Problem Spaces, Search Strategies (LU I & II).
- Problem Formulation:
    * State: Tuple (x, y) representing current volume in Jug A and Jug B.
    * Capacities: C_A (e.g. 4L) and C_B (e.g. 3L).
    * Initial State: (0, 0)
    * Goal State: Any state where x == Target or y == Target.
    * Actions (6 operators): Fill A, Fill B, Empty A, Empty B, Pour A->B, Pour B->A.
    * Path Cost: Unit step cost (1 move per action).
    * Solvability Theorem: By Bézout's Identity, Target is reachable iff
      Target <= max(C_A, C_B) and Target % gcd(C_A, C_B) == 0.

Authors: AI Micro-Project Team
"""

import math
import time
from collections import deque
import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class WaterJugResult:
    """Encapsulates the solution path, moves, and metrics for the Water Jug Problem."""
    success: bool
    status: str
    path: List[Tuple[int, int]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    cost: int = 0
    nodes_expanded: int = 0
    execution_time_ms: float = 0.0
    message: str = ""


def is_water_jug_solvable(cap_a: int, cap_b: int, target: int) -> Tuple[bool, str]:
    """
    Check if the Water Jug configuration is mathematically solvable using Bézout's Identity.
    """
    if cap_a <= 0 or cap_b <= 0 or target <= 0:
        return False, "Capacities and target volume must be positive integers."
    if target > max(cap_a, cap_b):
        return False, f"Target ({target}L) exceeds the maximum jug capacity ({max(cap_a, cap_b)}L)."
    
    g = math.gcd(cap_a, cap_b)
    if target % g != 0:
        return False, f"Target ({target}L) cannot be measured because it is not a multiple of gcd({cap_a}, {cap_b}) = {g}."
    
    return True, "Solvable"


def get_jug_successors(state: Tuple[int, int], cap_a: int, cap_b: int) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Generate all legal successor states from current volumes (x, y).

    Returns:
        List of tuples: (action_description, (new_x, new_y))
    """
    x, y = state
    moves = []

    # 1. Fill Jug A
    if x < cap_a:
        moves.append((f"Fill Jug A completely ({cap_a}L)", (cap_a, y)))

    # 2. Fill Jug B
    if y < cap_b:
        moves.append((f"Fill Jug B completely ({cap_b}L)", (x, cap_b)))

    # 3. Empty Jug A
    if x > 0:
        moves.append(("Empty Jug A onto the ground", (0, y)))

    # 4. Empty Jug B
    if y > 0:
        moves.append(("Empty Jug B onto the ground", (x, 0)))

    # 5. Pour Jug A into Jug B
    if x > 0 and y < cap_b:
        transfer = min(x, cap_b - y)
        moves.append((f"Pour {transfer}L from Jug A into Jug B", (x - transfer, y + transfer)))

    # 6. Pour Jug B into Jug A
    if y > 0 and x < cap_a:
        transfer = min(y, cap_a - x)
        moves.append((f"Pour {transfer}L from Jug B into Jug A", (x + transfer, y - transfer)))

    return moves


def solve_water_jug(
    cap_a: int = 4,
    cap_b: int = 3,
    target: int = 2,
    algo: str = "BFS"
) -> WaterJugResult:
    """
    Solve the Water Jug problem using Breadth-First Search (BFS) or A* Search.
    Guaranteed to return an optimal (shortest) sequence of water operations.
    """
    start_time = time.perf_counter()

    # Pre-check solvability
    solvable, reason = is_water_jug_solvable(cap_a, cap_b, target)
    if not solvable:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return WaterJugResult(
            success=False,
            status="UNSOLVABLE",
            execution_time_ms=elapsed_ms,
            message=reason
        )

    initial_state = (0, 0)
    # Edge case: target is 0
    if target == 0:
        return WaterJugResult(
            success=True,
            status="SUCCESS",
            path=[initial_state],
            actions=[],
            cost=0,
            nodes_expanded=0,
            execution_time_ms=0.0,
            message="Initial empty jugs already satisfy 0L target."
        )

    # Queue for BFS: stores state
    queue = deque([initial_state])
    parent_map: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], str]]] = {
        initial_state: None
    }

    nodes_expanded = 0

    while queue:
        curr = queue.popleft()
        nodes_expanded += 1

        # Goal check
        if curr[0] == target or curr[1] == target:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Reconstruct path
            path = []
            actions = []
            curr_node = curr
            while curr_node is not None:
                path.append(curr_node)
                entry = parent_map.get(curr_node)
                if entry is None or entry[0] is None:
                    break
                actions.append(entry[1])
                curr_node = entry[0]

            path.reverse()
            actions.reverse()

            return WaterJugResult(
                success=True,
                status="SUCCESS",
                path=path,
                actions=actions,
                cost=len(actions),
                nodes_expanded=nodes_expanded,
                execution_time_ms=elapsed_ms,
                message=f"Optimal solution found in {len(actions)} moves!"
            )

        for action_name, nxt_state in get_jug_successors(curr, cap_a, cap_b):
            if nxt_state not in parent_map:
                parent_map[nxt_state] = (curr, action_name)
                queue.append(nxt_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return WaterJugResult(
        success=False,
        status="NO_SOLUTION",
        nodes_expanded=nodes_expanded,
        execution_time_ms=elapsed_ms,
        message="Search space exhausted without reaching target."
    )
