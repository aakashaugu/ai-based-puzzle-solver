"""
puzzle.py - 8-Puzzle Problem Space Representation and Heuristics
================================================================
Academic Context:
- Artificial Intelligence Syllabus: Learning Unit I (Intelligent Agents & Problem-Solving)
  and Learning Unit II (Problem Spaces, Uninformed & Informed Search, Heuristics).
- Problem Formulation:
    * State Space: All permutations of the 9 tiles (3x3 grid) with numbers 0-8.
    * Initial State: Any valid configuration specified by user or generated.
    * Goal State: (1, 2, 3, 4, 5, 6, 7, 8, 0)
    * Actions: Moving the blank tile (0) Up, Down, Left, or Right.
    * Transition Model: State resulting from swapping 0 with adjacent tile.
    * Path Cost: Unit step cost (c(s, a, s') = 1).
    * Heuristic: Manhattan Distance (h1: sum of taxicab distances to goal positions).

Authors / Mentors: AI Micro-Project Team
"""

from typing import Tuple, List, Optional, Dict
import random

# Immutable Goal State representation
GOAL_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Precomputed goal coordinates (row, col) for each tile (0 to 8)
# In goal:
# Index 0: tile 1 -> (0, 0)
# Index 1: tile 2 -> (0, 1)
# Index 2: tile 3 -> (0, 2)
# Index 3: tile 4 -> (1, 0)
# Index 4: tile 5 -> (1, 1)
# Index 5: tile 6 -> (1, 2)
# Index 6: tile 7 -> (2, 0)
# Index 7: tile 8 -> (2, 1)
# Index 8: tile 0 -> (2, 2)
GOAL_POSITIONS: Dict[int, Tuple[int, int]] = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 0: (2, 2)
}

# Legal directional offsets for the blank space (row_offset, col_offset, action_name)
DIRECTIONS = [
    (-1, 0, 'Up'),     # Blank moves UP (tile above moves down)
    (1, 0, 'Down'),    # Blank moves DOWN (tile below moves up)
    (0, -1, 'Left'),   # Blank moves LEFT (tile to left moves right)
    (0, 1, 'Right'),   # Blank moves RIGHT (tile to right moves left)
]


def is_valid_board(state: Tuple[int, ...]) -> bool:
    """
    Validate that the given state contains exactly integers 0 through 8,
    each appearing once in a 9-element sequence.

    Args:
        state: State tuple to validate.

    Returns:
        True if the state is mathematically a valid 3x3 puzzle board; False otherwise.
    """
    if not isinstance(state, (tuple, list)):
        return False
    if len(state) != 9:
        return False
    # Check that all elements from 0 to 8 are present exactly once
    return set(state) == set(range(9))


def count_inversions(state: Tuple[int, ...]) -> int:
    """
    Count the number of inversions in the state, ignoring the blank tile (0).

    An inversion is a pair of tiles (a, b) such that a appears before b,
    but a > b. In the solved goal state (1, 2, 3, 4, 5, 6, 7, 8, 0),
    there are 0 inversions.

    Args:
        state: 9-element tuple representing the board.

    Returns:
        The total count of inversions.
    """
    tiles = [tile for tile in state if tile != 0]
    inversions = 0
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state: Tuple[int, ...]) -> bool:
    """
    Determine if an 8-puzzle board configuration is solvable with respect
    to the standard goal state: (1, 2, 3, 4, 5, 6, 7, 8, 0).

    Theorem:
    For an N x N puzzle where N is odd (N = 3):
    - A horizontal move swaps the blank with an adjacent tile in the sequence,
      which does not change the relative order of any two non-blank tiles.
    - A vertical move shifts a tile across exactly 2 intermediate tiles,
      changing the number of inversions by +2, -2, or 0 (an even difference).
    - Therefore, the parity of the inversion count is INVARIANT under all legal moves.
    - Since the goal state has 0 inversions (even parity), ANY reachable state
      MUST have an EVEN number of inversions.
    - Conversely, any state with an ODD number of inversions belongs to the
      disjoint partition and can never reach the goal state.

    Args:
        state: 9-element board state.

    Returns:
        True if solvable; False if mathematically impossible.
    """
    if not is_valid_board(state):
        return False
    return count_inversions(state) % 2 == 0


def is_goal(state: Tuple[int, ...]) -> bool:
    """Check if the given state matches the GOAL_STATE."""
    return state == GOAL_STATE


def manhattan_distance(state: Tuple[int, ...]) -> int:
    """
    Calculate the Manhattan Distance (Taxicab distance) heuristic:
        h(n) = sum_{tile=1..8} (|r_current - r_goal| + |c_current - c_goal|)

    Excludes the blank tile (0), in accordance with standard AI literature
    (Russell & Norvig, 'Artificial Intelligence: A Modern Approach').

    Properties:
    - Admissible: Never overestimates the true cost to reach the goal,
      because each move can decrease the Manhattan distance of at most one tile by 1.
    - Consistent (Monotonic): For any node n and successor n' via action a,
      h(n) <= c(n, a, n') + h(n') because |h(n) - h(n')| <= 1 = c(n, a, n').

    Args:
        state: 9-element board state.

    Returns:
        Sum of Manhattan distances for tiles 1 through 8.
    """
    total_distance = 0
    for idx, tile in enumerate(state):
        if tile != 0:
            current_r, current_c = divmod(idx, 3)
            goal_r, goal_c = GOAL_POSITIONS[tile]
            total_distance += abs(current_r - goal_r) + abs(current_c - goal_c)
    return total_distance


def misplaced_tiles(state: Tuple[int, ...]) -> int:
    """
    Calculate the Misplaced Tiles (Hamming Distance) heuristic:
        h_misplaced(n) = count of tiles t in 1..8 where t != GOAL_STATE[idx]

    Excludes the blank tile (0).
    Admissible, but generally dominated by Manhattan distance.

    Args:
        state: 9-element board state.

    Returns:
        Count of misplaced tiles.
    """
    count = 0
    for idx, tile in enumerate(state):
        if tile != 0 and tile != GOAL_STATE[idx]:
            count += 1
    return count


def get_legal_moves(state: Tuple[int, ...]) -> List[Tuple[str, int, Tuple[int, ...]]]:
    """
    Generate all legal successor states from the current board.

    Successors are generated by moving the blank space (0) Up, Down, Left, or Right.

    Args:
        state: Current 9-element board state.

    Returns:
        List of tuples: (action_name, moved_tile_value, successor_state_tuple)
        Example: ('Up', 4, (1, 2, 3, 0, 5, 6, 7, 8, 4))
    """
    blank_idx = state.index(0)
    blank_r, blank_c = divmod(blank_idx, 3)
    successors = []

    for dr, dc, action in DIRECTIONS:
        new_r, new_c = blank_r + dr, blank_c + dc
        if 0 <= new_r < 3 and 0 <= new_c < 3:
            target_idx = new_r * 3 + new_c
            moved_tile = state[target_idx]

            # Construct new state with swapped tiles
            state_list = list(state)
            state_list[blank_idx], state_list[target_idx] = state_list[target_idx], state_list[blank_idx]
            successor_state = tuple(state_list)

            successors.append((action, moved_tile, successor_state))

    return successors


def generate_random_solvable_puzzle(steps: Optional[int] = None) -> Tuple[int, ...]:
    """
    Generate a verified solvable random 8-puzzle board.

    Args:
        steps: If provided, generates the puzzle by taking `steps` random moves
               from the goal state (useful for controlling difficulty).
               If None, generates a random permutation of 0-8 with even inversions.

    Returns:
        A solvable 9-element tuple representing the board.
    """
    if steps is not None and steps > 0:
        current = GOAL_STATE
        last_action = None
        # Reverse mapping for anti-backtracking
        opposite = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}

        for _ in range(steps):
            moves = get_legal_moves(current)
            # Filter out immediate reversal if multiple moves are available
            non_reversing = [m for m in moves if m[0] != opposite.get(last_action, '')]
            chosen = random.choice(non_reversing if non_reversing else moves)
            last_action = chosen[0]
            current = chosen[2]
        return current

    # Pure random permutation with solvable parity
    while True:
        state_list = list(range(9))
        random.shuffle(state_list)
        candidate = tuple(state_list)
        if is_solvable(candidate) and candidate != GOAL_STATE:
            return candidate


def format_board(state: Tuple[int, ...]) -> str:
    """Format the 9-element tuple into a readable 3x3 grid string."""
    rows = []
    for r in range(3):
        row_str = " ".join(str(state[r * 3 + c]) if state[r * 3 + c] != 0 else "_" for c in range(3))
        rows.append(row_str)
    return "\n".join(rows)


def parse_board(text_or_list) -> Optional[Tuple[int, ...]]:
    """
    Parse a user input (string or list) into a 9-element tuple.

    Returns None if parsing fails or invalid board.
    """
    try:
        if isinstance(text_or_list, (tuple, list)):
            nums = [int(x) for x in text_or_list]
        elif isinstance(text_or_list, str):
            # Accept space or comma separated numbers
            clean_str = text_or_list.replace(',', ' ').replace('\n', ' ')
            nums = [int(tok.strip()) for tok in clean_str.split() if tok.strip()]
        else:
            return None

        if len(nums) == 9 and is_valid_board(tuple(nums)):
            return tuple(nums)
        return None
    except (ValueError, TypeError):
        return None


# Curated Presets with verified optimal move distances for academic demonstrations
PRESET_PUZZLES: Dict[str, Dict[str, any]] = {
    "Easy (2 Moves)": {
        "state": (1, 2, 0, 4, 5, 3, 7, 8, 6),
        "difficulty": "Easy",
        "description": "Requires exactly 2 optimal moves. Ideal for quick smoke tests and verifying equivalence."
    },
    "Easy (4 Moves)": {
        "state": (1, 5, 2, 4, 0, 3, 7, 8, 6),
        "difficulty": "Easy",
        "description": "Requires exactly 4 optimal moves. Quick step-by-step playback."
    },
    "Easy (6 Moves)": {
        "state": (1, 5, 2, 4, 8, 3, 0, 7, 6),
        "difficulty": "Easy",
        "description": "Requires exactly 6 optimal moves. Both BFS and A* solve almost instantaneously."
    },
    "Medium (8 Moves)": {
        "state": (0, 5, 2, 1, 8, 3, 4, 7, 6),
        "difficulty": "Medium",
        "description": "Requires exactly 8 optimal moves. Demonstrates A* expanding ~9 nodes vs BFS expanding ~155 nodes."
    },
    "Medium (12 Moves)": {
        "state": (5, 8, 2, 1, 7, 3, 0, 4, 6),
        "difficulty": "Medium",
        "description": "Requires exactly 12 optimal moves. Dramatic divergence between informed and uninformed search."
    },
    "Hard (16 Moves)": {
        "state": (8, 7, 2, 5, 0, 3, 1, 4, 6),
        "difficulty": "Hard",
        "description": "Requires exactly 16 optimal moves. A* solves in milliseconds; BFS explores thousands of nodes."
    },
    "Challenge (20 Moves)": {
        "state": (8, 7, 0, 5, 4, 2, 1, 6, 3),
        "difficulty": "Challenge",
        "description": "Requires exactly 20 optimal moves. Real-world stress test highlighting heuristic guidance."
    },
    "Already Solved (0 Moves)": {
        "state": GOAL_STATE,
        "difficulty": "Trivial",
        "description": "Goal state: Tests zero-move boundary handling and immediate termination."
    },
    "Unsolvable (Odd Parity Demo)": {
        "state": (2, 1, 3, 4, 5, 6, 7, 8, 0),
        "difficulty": "Impossible",
        "description": "Tiles 1 and 2 are swapped, giving 1 inversion (odd parity). Mathematically impossible to reach goal."
    }
}
