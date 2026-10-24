"""
Unit tests for algorithms.py
============================
Tests BFS, DFS, and A* search implementations:
- Already-solved puzzle returns 0-move solution
- Every consecutive pair of states differs by exactly 1 legal move
- A* returns shortest solution for selected test cases
- BFS returns shortest solution for small solvable test cases
- A* and BFS produce identical optimal solution length on solvable puzzles
- DFS respects its expansion limit
- Handles unsolvable puzzles gracefully without crashing
Compatible with both pytest and standard python -m unittest.
"""

import sys
import os
import unittest

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puzzle import (
    GOAL_STATE,
    is_goal,
    is_valid_board,
    get_legal_moves,
    PRESET_PUZZLES
)
from algorithms import (
    bfs_solve,
    dfs_solve,
    astar_solve,
    SearchResult
)


class TestAlgorithms(unittest.TestCase):
    """Test suite for AI search algorithms."""

    def verify_solution_path_validity(self, result: SearchResult, initial_state):
        """
        Helper function to verify that:
        1. The path starts with the initial state and ends with the goal state.
        2. Every consecutive pair of states in the path differs by exactly one legal move.
        3. The number of actions equals len(path) - 1.
        4. Recorded action and moved tile match the actual state transition.
        """
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.path), 1)
        self.assertEqual(result.path[0], initial_state)
        self.assertEqual(result.path[-1], GOAL_STATE)
        self.assertEqual(len(result.actions), len(result.path) - 1)
        self.assertEqual(len(result.moved_tiles), len(result.path) - 1)
        self.assertEqual(result.cost, len(result.actions))

        for i in range(len(result.path) - 1):
            s_curr = result.path[i]
            s_next = result.path[i + 1]
            action = result.actions[i]
            tile = result.moved_tiles[i]

            # Verify that s_next is a legal successor of s_curr
            legal = get_legal_moves(s_curr)
            matches = [m for m in legal if m[2] == s_next]
            self.assertEqual(len(matches), 1, f"State transition at step {i} is not a legal move!")
            legal_action, legal_tile, _ = matches[0]
            self.assertEqual(legal_action, action)
            self.assertEqual(legal_tile, tile)

    def test_already_solved_puzzle(self):
        """Verify that an already-solved puzzle returns a zero-move solution across all algorithms."""
        for algo_func in [astar_solve, bfs_solve, dfs_solve]:
            res = algo_func(GOAL_STATE)
            self.assertTrue(res.success)
            self.assertEqual(res.cost, 0)
            self.assertEqual(len(res.path), 1)
            self.assertEqual(res.path[0], GOAL_STATE)
            self.assertEqual(len(res.actions), 0)
            self.assertEqual(res.nodes_expanded, 0)

    def test_astar_finds_optimal_path_easy(self):
        """Verify that A* finds the optimal path for a 2-move puzzle and verifies step validity."""
        start_state = PRESET_PUZZLES["Easy (2 Moves)"]["state"]
        res = astar_solve(start_state)
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 2)
        self.verify_solution_path_validity(res, start_state)

    def test_bfs_finds_optimal_path_easy(self):
        """Verify that BFS finds the optimal path for a 2-move puzzle."""
        start_state = PRESET_PUZZLES["Easy (2 Moves)"]["state"]
        res = bfs_solve(start_state)
        self.assertTrue(res.success)
        self.assertEqual(res.cost, 2)
        self.verify_solution_path_validity(res, start_state)

    def test_astar_bfs_optimality_equivalence(self):
        """Verify that A* and BFS produce the exact same minimal solution length on 4-move puzzle."""
        start_state = PRESET_PUZZLES["Easy (4 Moves)"]["state"]
        res_astar = astar_solve(start_state)
        res_bfs = bfs_solve(start_state)

        self.assertTrue(res_astar.success)
        self.assertTrue(res_bfs.success)
        # Both must find the minimal path length
        self.assertEqual(res_astar.cost, res_bfs.cost)
        self.assertEqual(res_astar.cost, 4)
        # A* expands fewer or equal nodes compared to BFS
        self.assertLessEqual(res_astar.nodes_expanded, res_bfs.nodes_expanded)
        self.verify_solution_path_validity(res_astar, start_state)
        self.verify_solution_path_validity(res_bfs, start_state)

    def test_dfs_respects_expansion_limit(self):
        """Verify that DFS halts safely when expansion ceiling is reached without crashing."""
        medium_state = PRESET_PUZZLES["Medium (8 Moves)"]["state"]
        low_limit = 100
        res = dfs_solve(medium_state, max_expansions=low_limit)

        # Should hit limit and not exceed limit + 1
        self.assertLessEqual(res.nodes_expanded, low_limit + 1)
        if not res.success:
            self.assertEqual(res.status, "LIMIT_REACHED")
            self.assertIn("limit", res.termination_reason.lower())

    def test_unsolvable_puzzle_handled_safely(self):
        """Verify that all algorithms detect unsolvable puzzles cleanly via parity without endless search."""
        unsolvable = PRESET_PUZZLES["Unsolvable (Odd Parity Demo)"]["state"]
        for algo_func in [astar_solve, bfs_solve, dfs_solve]:
            res = algo_func(unsolvable)
            self.assertFalse(res.success)
            self.assertEqual(res.status, "UNSOLVABLE")
            self.assertIn("unsolvable", res.termination_reason.lower())


if __name__ == '__main__':
    unittest.main()
