"""
Unit tests for water_jug.py
===========================
Tests solvability check via Bézout's identity, successor generation,
and optimal BFS solution path for the Water Jug Problem.
"""

import sys
import os
import unittest

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from water_jug import (
    is_water_jug_solvable,
    get_jug_successors,
    solve_water_jug
)


class TestWaterJug(unittest.TestCase):
    """Test suite for Water Jug problem solving and rules."""

    def test_solvability_bezout(self):
        """Verify solvability conditions via gcd and Bézout's identity."""
        # Classic 4L and 3L to get 2L -> gcd(4,3)=1, 2%1==0 -> Solvable
        solvable, _ = is_water_jug_solvable(4, 3, 2)
        self.assertTrue(solvable)

        # 6L and 4L to get 3L -> gcd(6,4)=2, 3%2 != 0 -> Unsolvable
        unsolvable, msg = is_water_jug_solvable(6, 4, 3)
        self.assertFalse(unsolvable)
        self.assertIn("gcd", msg)

        # Target greater than capacity
        exceeds, msg2 = is_water_jug_solvable(4, 3, 5)
        self.assertFalse(exceeds)
        self.assertIn("exceeds", msg2)

    def test_solve_classic_4_3_2(self):
        """Verify optimal solution for classic (4L, 3L, 2L) Water Jug problem."""
        res = solve_water_jug(cap_a=4, cap_b=3, target=2)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "SUCCESS")
        self.assertGreater(res.cost, 0)
        # Check that final state has 2 in either jug
        final_state = res.path[-1]
        self.assertTrue(final_state[0] == 2 or final_state[1] == 2)
        # Verify consecutive steps
        for i in range(len(res.path) - 1):
            curr = res.path[i]
            nxt = res.path[i + 1]
            successors = [s[1] for s in get_jug_successors(curr, 4, 3)]
            self.assertIn(nxt, successors)

    def test_unsolvable_execution(self):
        """Verify that solving an unsolvable configuration returns clean status without searching."""
        res = solve_water_jug(cap_a=6, cap_b=4, target=3)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "UNSOLVABLE")


if __name__ == '__main__':
    unittest.main()
