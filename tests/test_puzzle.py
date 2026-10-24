"""
Unit tests for puzzle.py
========================
Tests validation, legal moves, solvability check (inversion parity),
heuristics (Manhattan distance), and random solvable generator.
Compatible with both pytest and standard python -m unittest.
"""

import sys
import os
import unittest

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puzzle import (
    GOAL_STATE,
    is_valid_board,
    count_inversions,
    is_solvable,
    is_goal,
    manhattan_distance,
    misplaced_tiles,
    get_legal_moves,
    generate_random_solvable_puzzle,
    parse_board,
    PRESET_PUZZLES
)


class TestPuzzle(unittest.TestCase):
    """Test suite for puzzle representation and rules."""

    def test_goal_state_properties(self):
        """Verify goal state definition, validity, and that its heuristic values are zero."""
        self.assertTrue(is_valid_board(GOAL_STATE))
        self.assertTrue(is_goal(GOAL_STATE))
        self.assertTrue(is_solvable(GOAL_STATE))
        self.assertEqual(count_inversions(GOAL_STATE), 0)
        # Academic requirement: Goal state has Manhattan distance of zero
        self.assertEqual(manhattan_distance(GOAL_STATE), 0)
        self.assertEqual(misplaced_tiles(GOAL_STATE), 0)

    def test_invalid_boards_rejected(self):
        """Validate that invalid boards with wrong lengths, numbers, or duplicates fail validation."""
        # Incorrect length
        self.assertFalse(is_valid_board((1, 2, 3, 4, 5, 6, 7, 8)))
        self.assertFalse(is_valid_board((1, 2, 3, 4, 5, 6, 7, 8, 0, 9)))

        # Duplicate values
        self.assertFalse(is_valid_board((1, 1, 2, 3, 4, 5, 6, 7, 0)))
        self.assertFalse(is_valid_board((0, 0, 2, 3, 4, 5, 6, 7, 8)))

        # Out of range numbers
        self.assertFalse(is_valid_board((1, 2, 3, 4, 5, 6, 7, 8, 9)))
        self.assertFalse(is_valid_board((-1, 1, 2, 3, 4, 5, 6, 7, 8)))

        # Non-iterable or invalid types
        self.assertFalse(is_valid_board("invalid"))
        self.assertFalse(is_valid_board(123456780))

    def test_solvability_classification(self):
        """Verify that solvable and unsolvable boards are classified correctly using inversion parity."""
        # Solvable state: Easy 2 moves preset
        easy_state = PRESET_PUZZLES["Easy (2 Moves)"]["state"]
        self.assertTrue(is_solvable(easy_state))
        self.assertEqual(count_inversions(easy_state) % 2, 0)

        # Unsolvable state: Goal with tiles 1 and 2 swapped
        # (2, 1, 3, 4, 5, 6, 7, 8, 0) has exactly 1 inversion (2 > 1) -> odd -> unsolvable
        unsolvable_state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
        self.assertTrue(is_valid_board(unsolvable_state))
        self.assertEqual(count_inversions(unsolvable_state), 1)
        self.assertFalse(is_solvable(unsolvable_state))

        # Another unsolvable state: 7 and 8 swapped
        unsolvable_state_2 = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        self.assertEqual(count_inversions(unsolvable_state_2), 1)
        self.assertFalse(is_solvable(unsolvable_state_2))

        # Known solvable: (1, 8, 2, 0, 4, 3, 7, 6, 5)
        solvable_state = (1, 8, 2, 0, 4, 3, 7, 6, 5)
        self.assertTrue(is_solvable(solvable_state))

    def test_legal_moves_boundaries(self):
        """Verify that legal moves never move the blank tile outside the 3x3 board."""
        # Blank in corner (top-left index 0: row 0, col 0)
        corner_top_left = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        moves_tl = get_legal_moves(corner_top_left)
        action_names_tl = [m[0] for m in moves_tl]
        # Can only move Down or Right
        self.assertEqual(set(action_names_tl), {'Down', 'Right'})
        self.assertEqual(len(moves_tl), 2)

        # Blank in bottom-right corner (index 8: row 2, col 2)
        moves_br = get_legal_moves(GOAL_STATE)
        action_names_br = [m[0] for m in moves_br]
        # Can only move Up or Left
        self.assertEqual(set(action_names_br), {'Up', 'Left'})
        self.assertEqual(len(moves_br), 2)

        # Blank in center (index 4: row 1, col 1)
        center_blank = (1, 2, 3, 4, 0, 5, 6, 7, 8)
        moves_c = get_legal_moves(center_blank)
        action_names_c = [m[0] for m in moves_c]
        # Can move all 4 directions
        self.assertEqual(set(action_names_c), {'Up', 'Down', 'Left', 'Right'})
        self.assertEqual(len(moves_c), 4)

        # Check for all possible states that every successor is valid and has exactly 9 elements
        for _, moved_tile, successor in moves_c:
            self.assertTrue(is_valid_board(successor))
            self.assertEqual(successor.count(0), 1)
            self.assertIn(moved_tile, range(1, 9))

    def test_manhattan_distance_values(self):
        """Verify Manhattan distance calculation against known manual calculations."""
        # State: 1 tile misplaced by 1 step
        state_1_step = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        self.assertEqual(manhattan_distance(state_1_step), 1)

        # State: (2, 8, 3, 1, 6, 4, 7, 0, 5) -> manual total is 9
        calc_state = (2, 8, 3, 1, 6, 4, 7, 0, 5)
        self.assertEqual(manhattan_distance(calc_state), 9)

    def test_random_solvable_generator(self):
        """Verify that random puzzle generator always produces solvable boards."""
        for _ in range(10):
            puzzle = generate_random_solvable_puzzle()
            self.assertTrue(is_valid_board(puzzle))
            self.assertTrue(is_solvable(puzzle))

        # Test random walk with steps
        for steps in [3, 8, 15]:
            walk_puzzle = generate_random_solvable_puzzle(steps=steps)
            self.assertTrue(is_valid_board(walk_puzzle))
            self.assertTrue(is_solvable(walk_puzzle))

    def test_parse_board(self):
        """Verify parsing of various user string inputs."""
        self.assertEqual(parse_board("1 2 3 4 5 6 7 8 0"), GOAL_STATE)
        self.assertEqual(parse_board("1,2,3,4,5,6,7,8,0"), GOAL_STATE)
        self.assertEqual(parse_board("1 2 3\n4 5 6\n7 8 0"), GOAL_STATE)
        self.assertIsNone(parse_board("invalid text"))
        self.assertEqual(parse_board([1, 2, 3, 4, 5, 6, 7, 8, 0]), GOAL_STATE)


if __name__ == '__main__':
    unittest.main()
