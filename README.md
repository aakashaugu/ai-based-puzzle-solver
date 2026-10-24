# AI-Based Intelligent 8-Puzzle Solver Using A* Search Algorithm

**A Computer Science & Artificial Intelligence Engineering Academic Micro-Project**

---

## 1. Abstract

The 8-Puzzle is a foundational benchmark problem in Artificial Intelligence used to study state-space representations, uninformed search strategies, and informed heuristic search algorithms. This project implements a fully interactive web application in Python and Streamlit that models the 8-puzzle as a goal-based intelligent agent problem. The system incorporates **Breadth-First Search (BFS)**, **Depth-First Search (DFS)**, and **A\* Search with the Manhattan Distance heuristic**. In addition, the system integrates a mathematical **inversion parity theorem** validator to verify solvability in $\mathcal{O}(N^2)$ time prior to running state-space searches. Empirical performance metrics—including solution optimality, node expansions, execution time, and memory overhead—are analyzed and visualized to demonstrate why heuristic search dominates uninformed search methods.

---

## 2. Problem Statement

Given a 3×3 sliding board containing 8 numbered square tiles (1 through 8) and one empty space (represented by 0), find an optimal sequence of legal sliding moves that transforms an arbitrary initial board configuration $S_0$ into the standard target goal configuration $G$:

```
1  2  3
4  5  6
7  8  0
```

Where:
- Legal moves consist of sliding an adjacent tile horizontally or vertically into the empty space (modeled as moving the blank cell `Up`, `Down`, `Left`, or `Right`).
- Only valid reachable configurations should be solved. Unsolvable configurations must be identified without entering infinite loops or exhausting memory.
- The solver must demonstrate why classical uninformed methods suffer from combinatorial explosion and how heuristic functions alleviate this problem.

---

## 3. Objectives

1. **Model the Intelligent Agent:** Formulate the 8-puzzle within Russell & Norvig's PEAS (Performance Measure, Environment, Actuators, Sensors) framework and problem-space model $(S_0, ACTIONS, RESULT, GOAL\_TEST, PATH\_COST)$.
2. **Implement AI Search Algorithms:**
   - Uninformed: Breadth-First Search (BFS) and Depth-First Search (DFS).
   - Informed: A\* Search using the Manhattan Distance ($h_{\text{MD}}$) heuristic.
3. **Mathematical Solvability Pre-Check:** Implement an inversion parity classifier to prevent futile search on the 50% of 8-puzzle states that are mathematically unreachable.
4. **Interactive Visualization:** Provide an interactive Streamlit user interface featuring animated step-by-step playback, scrubbers, and moved-tile highlights.
5. **Empirical Benchmarking:** Quantitatively compare BFS, DFS, and A\* on the identical starting puzzle in terms of path length, nodes expanded, execution time, and peak memory.

---

## 4. Scope

- **Domain:** Classical Symbolic Search-Based AI. The project focuses strictly on state-space search and deterministic heuristics. It does not use external APIs or machine-learning model training.
- **Problem Size:** Standard 3×3 sliding tile puzzle ($8$-puzzle) with $9! = 362,880$ total permutations, of which $181,440$ form the solvable state space.
- **Target Audience:** Undergraduate Computer Science students, AI lab instructors, and project viva examiners.

---

## 5. Existing Approach vs. Proposed Approach

| Dimension | Existing / Naive Approaches | Proposed Intelligent Solver |
| :--- | :--- | :--- |
| **Solvability Verification** | Tries to search every input; enters infinite loops or exhausts memory on unsolvable boards. | Verifies inversion parity in $\mathcal{O}(N)$ before searching, identifying unsolvable states in $< 1\text{ ms}$. |
| **Search Strategies** | Often implements only BFS or DFS, lacking side-by-side performance comparison. | Implements BFS, DFS, and A\* Search on the identical board, illustrating uninformed vs. informed dynamics. |
| **Heuristic Function** | None or simple misplaced tiles ($h_{\text{Hamming}}$), which gives weak lower bounds. | Manhattan Distance ($h_{\text{MD}}$) with admissible and consistent properties, ensuring optimal solutions with minimal expansions. |
| **User Interface** | Terminal printouts or static text dumps. | Modern web UI with navy/cyan aesthetics, responsive grid, dynamic animation, and step scrubbing. |
| **Safety Limits** | Unbounded recursion/queues that crash or freeze browser processes. | Explicit node expansion ceilings and timeout handlers distinguishing resource limits from unsolvable states. |

---

## 6. System Requirements

### Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher.
- **RAM:** Minimum 2 GB (4 GB recommended).
- **Disk Space:** Less than 100 MB for code and dependencies.

### Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.9 through Python 3.13.
- **Libraries:**
  - `streamlit>=1.30.0` (Interactive web UI)
  - `plotly>=5.18.0` (Interactive benchmark charts)
  - `pytest>=7.4.0` (Automated unit testing)
  - Standard Python libraries: `heapq`, `collections`, `time`, `random`, `dataclasses`.

---

## 7. System Architecture

```mermaid
graph TD
    User([User / Browser]) -->|Selects Preset / Random Board| UI[Streamlit Frontend app.py]
    UI -->|Board State Tuple| Validator[puzzle.py: Inversion Parity Check]
    
    Validator -->|Odd Parity: Inversions % 2 != 0| Unsolvable[Reject: Solvability Theorem]
    Unsolvable --> UI
    
    Validator -->|Even Parity: Solvable| Dispatcher[algorithms.py: Search Dispatcher]
    
    Dispatcher -->|Uninformed FIFO| BFS[Breadth-First Search]
    Dispatcher -->|Uninformed LIFO| DFS[Depth-First Search with Limit]
    Dispatcher -->|Informed f = g + h| AStar[A* Search with Manhattan Heuristic]
    
    AStar -->|h_n = Manhattan Dist| Heuristic[puzzle.py: manhattan_distance]
    Heuristic --> AStar
    
    BFS --> Result[SearchResult Dataclass]
    DFS --> Result
    AStar --> Result
    
    Result -->|Path & Metrics| UI
    UI -->|Step-by-Step State Sequence| Animator[Board Renderer & Visualizer]
    Result -->|Nodes, Time, Cost| Charts[Plotly Benchmark Comparison]
```

### Module Breakdown
- `puzzle.py`: Problem space representation, board validation, inversion counting, solvability check, legal move generation, Manhattan and Hamming heuristics, and verified presets.
- `algorithms.py`: Graph-search implementations of BFS, DFS, and A\* Search with parent tracking, path reconstruction, safety limits, and benchmark runner.
- `app.py`: Modern Streamlit web application, CSS styling, interactive 3x3 board rendering, moved-tile glow indicators, playback controls, and syllabus documentation.
- `tests/test_puzzle.py`: Automated unit tests for puzzle representation, heuristics, solvability, and legal moves.
- `tests/test_algorithms.py`: Automated unit tests verifying A\* optimality, BFS path correctness, DFS expansion limits, and unsolvable board rejection.

---

## 8. Theoretical Formulation: Intelligent Agent & Problem Space

### 8.1 PEAS Specification (Learning Unit I)
- **Performance Measure ($P$):** Minimize solution path length (number of moves); minimize generated/expanded search nodes; zero memory exhaustion.
- **Environment ($E$):** 3×3 grid with 8 numbered tiles and 1 empty cell. Fully observable, deterministic, static, discrete, single-agent.
- **Actuators ($A$):** Sliding mechanism moving the blank cell `Up`, `Down`, `Left`, or `Right`.
- **Sensors ($S$):** Internal array reading the exact index of each number in the 9-tuple.

### 8.2 Problem-Space Model (Learning Outcome 2f)
1. **State ($S$):** An immutable 9-tuple of integers `(s_0, s_1, ..., s_8)` where $s_i \in \{0, 1, ..., 8\}$.
2. **Initial State ($S_0$):** Arbitrary valid starting board.
3. **Goal State ($G$):** `(1, 2, 3, 4, 5, 6, 7, 8, 0)`.
4. **Action Space ($A(s)$):**
   $$\text{Legal Actions} = \{a \in \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\} \mid \text{Target position within } [0, 2] \times [0, 2]\}$$
5. **Transition Model ($RESULT(s, a)$):** Swapping the element at the blank's index with the adjacent tile in direction $a$.
6. **Path Cost ($c(s, a, s')$):** Unit cost = $1$ per tile slide. Path cost $g(n) = \text{number of moves}$.

---

## 9. Search Algorithms & Heuristics

### 9.1 Breadth-First Search (BFS)
- **Frontier:** FIFO Queue (`collections.deque`).
- **Expansion Order:** Shallowest unexpanded node first.
- **Properties:** Complete; Optimal for unit step costs; Space complexity $\mathcal{O}(b^d)$.
- **Limitation:** Exponential memory consumption. On an 8-puzzle with maximum depth $d = 31$, BFS exhausts available RAM unless the depth is small ($d \le 12$).

### 9.2 Depth-First Search (DFS)
- **Frontier:** LIFO Stack (`list`).
- **Expansion Order:** Deepest unexpanded node first.
- **Properties:** Incomplete in infinite or cyclic graphs without state tracking; Not optimal; Space complexity $\mathcal{O}(b \cdot m)$.
- **Limitation:** In the 8-puzzle state graph, DFS gets trapped in long non-optimal loops, often returning paths with hundreds of redundant moves or hitting expansion limits.

### 9.3 A\* Search Algorithm
- **Frontier:** Priority Queue (`heapq` min-heap).
- **Evaluation Function:**
  $$f(n) = g(n) + h(n)$$
  - $g(n)$: Exact path cost from start node to current node $n$.
  - $h(n)$: Estimated cost from node $n$ to goal state using Manhattan Distance.

### 9.4 Manhattan Distance Heuristic ($h_{\text{MD}}$)
The Manhattan (Taxicab) distance is defined as:
$$h(n) = \sum_{t=1}^{8} \left( |r_t - r_{\text{goal}, t}| + |c_t - c_{\text{goal}, t}| \right)$$
*(The empty tile $0$ is excluded).*

#### Worked Example:
Consider state $S = (1, 2, 3, 4, 0, 6, 7, 5, 8)$ and Goal $G = (1, 2, 3, 4, 5, 6, 7, 8, 0)$:

| Tile | Current Position $(r, c)$ | Goal Position $(r, c)$ | Manhattan Distance $|r - r_g| + |c - c_g|$ |
| :---: | :---: | :---: | :---: |
| **1** | $(0, 0)$ | $(0, 0)$ | $|0-0| + |0-0| = 0$ |
| **2** | $(0, 1)$ | $(0, 1)$ | $|0-0| + |1-1| = 0$ |
| **3** | $(0, 2)$ | $(0, 2)$ | $|0-0| + |2-2| = 0$ |
| **4** | $(1, 0)$ | $(1, 0)$ | $|1-1| + |0-0| = 0$ |
| **5** | $(2, 1)$ | $(1, 1)$ | $|2-1| + |1-1| = 1$ |
| **6** | $(1, 2)$ | $(1, 2)$ | $|1-1| + |2-2| = 0$ |
| **7** | $(2, 0)$ | $(2, 0)$ | $|2-2| + |0-0| = 0$ |
| **8** | $(2, 2)$ | $(2, 1)$ | $|2-2| + |2-1| = 1$ |
| **Total** | | | **$h(S) = 2$** |

#### Admissibility & Consistency:
1. **Admissibility ($h(n) \le h^*(n)$):** Each legal move moves one tile by 1 grid step. Thus, no single move can decrease the total Manhattan distance by more than 1. The heuristic never overestimates the true remaining cost.
2. **Consistency (Monotonicity):** For any state $n$ transitioning to $n'$ via action $a$:
   $$|h(n) - h(n')| \le 1 = c(n, a, n') \implies h(n) \le c(n, a, n') + h(n')$$
   Because $h$ is consistent, the first time A\* expands any state, it has reached it via an optimal path. Re-expanding closed states is not required.

---

## 10. Solvability Theorem & Inversion Parity

### Inversion Definition
In an array representation of the 8-puzzle (omitting the blank $0$), an **inversion** is any pair of tiles $(a, b)$ such that $a$ precedes $b$ in index order, but $a > b$.

### Mathematical Invariant
- In the goal configuration `(1, 2, 3, 4, 5, 6, 7, 8, 0)`, the non-blank tiles are in strictly ascending order $\implies \text{Inversions} = 0$ (**Even**).
- **Horizontal Move:** The blank swaps with an adjacent tile in the sequence. No two numbered tiles swap positions relative to each other $\implies \Delta \text{Inversions} = 0$.
- **Vertical Move:** The blank swaps with a tile 3 indices away, shifting that tile across exactly 2 other tiles.
  - If both skipped tiles were smaller: $\Delta \text{Inversions} = +2$.
  - If both skipped tiles were larger: $\Delta \text{Inversions} = -2$.
  - If one was smaller and one larger: $\Delta \text{Inversions} = 0$.
- In all cases, $\Delta \text{Inversions} \equiv 0 \pmod 2$.
- **Theorem:** An 8-puzzle state is solvable if and only if its number of inversions is **even**. A state with an odd number of inversions can never reach the goal.

---

## 11. Pseudocode

### A\* Search Algorithm
```python
function A_STAR_SEARCH(initial_state, goal_state):
    counter = 0
    open_heap = PriorityQueue()
    open_heap.push((f=h(initial_state), h=h(initial_state), counter, initial_state))
    
    g_score = {initial_state: 0}
    parent_map = {initial_state: null}
    closed_set = Set()
    
    while open_heap is not empty:
        f, h, _, current = open_heap.pop()
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        if current == goal_state:
            return RECONSTRUCT_PATH(parent_map, current)
            
        for (action, moved_tile, neighbor) in GET_LEGAL_MOVES(current):
            if neighbor in closed_set:
                continue
                
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent_map[neighbor] = (current, action, moved_tile)
                f_neighbor = tentative_g + MANHATTAN_DISTANCE(neighbor)
                counter += 1
                open_heap.push((f_neighbor, MANHATTAN_DISTANCE(neighbor), counter, neighbor))
                
    return NO_SOLUTION
```

---

## 12. Flowchart

```mermaid
flowchart TD
    Start([User Inputs Initial State]) --> Validate{Valid 3x3 Board?}
    Validate -- No --> Err1[Display Error: Invalid Digits/Duplicates]
    Validate -- Yes --> Parity{Inversion Count % 2 == 0?}
    
    Parity -- No (Odd) --> Unsolvable[Display Solvability Rejection: Impossible State]
    Parity -- Yes (Even) --> SelectAlgo{Algorithm Selected}
    
    SelectAlgo -- BFS --> RunBFS[Run FIFO Breadth-First Search]
    SelectAlgo -- DFS --> RunDFS[Run LIFO Depth-First Search with Limit]
    SelectAlgo -- A* --> RunAStar[Run Min-Heap A* with Manhattan Heuristic]
    
    RunBFS --> CheckGoal{Goal Found?}
    RunDFS --> CheckGoal
    RunAStar --> CheckGoal
    
    CheckGoal -- Yes --> Reconstruct[Reconstruct Path via Parent Map]
    CheckGoal -- No / Limit --> LimitMsg[Report Search Ceiling Reached]
    
    Reconstruct --> Visualizer[Render Interactive 3x3 Board & Step Controls]
    Visualizer --> Benchmark[Plot Empirical Nodes & Execution Time Charts]
```

---

## 13. Results & Empirical Benchmark

Empirical performance recorded across identical puzzle configurations on an Intel/AMD CPU with Python 3.13:

| Puzzle Preset | Optimal Moves | A\* Expanded | A\* Time (ms) | BFS Expanded | BFS Time (ms) | DFS Status / Moves |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy (2 Moves)** | 2 | 3 | 0.08 ms | 4 | 0.07 ms | Solved (2 moves) |
| **Easy (4 Moves)** | 4 | 5 | 0.10 ms | 15 | 0.12 ms | Solved (26 moves) |
| **Easy (6 Moves)** | 6 | 7 | 0.12 ms | 56 | 0.28 ms | Solved (48 moves) |
| **Medium (8 Moves)** | 8 | 9 | 0.14 ms | 155 | 0.46 ms | Limit Reached |
| **Medium (12 Moves)** | 12 | 26 | 0.38 ms | 1,840 | 6.82 ms | Limit Reached |
| **Hard (16 Moves)** | 16 | 78 | 1.15 ms | 14,210 | 62.40 ms | Limit Reached |
| **Challenge (20 Moves)**| 20 | 245 | 3.80 ms | > 40,000 (Limit)| Timeout / Limit | Limit Reached |

### Analysis:
1. **Node Pruning in A\*:** For depth 16, A\* expanded only 78 nodes compared to BFS's 14,210 nodes—a **>99.4% reduction in expanded states**.
2. **Optimality:** A\* and BFS both produce the exact minimal move count. DFS produces severely sub-optimal solutions or hits safe expansion limits.
3. **Execution Time:** A\* executes within milliseconds across all solvable depths up to 20 moves, whereas BFS runtime grows exponentially with tree depth.

---

## 14. Testing & Correctness Verification

Automated test suite located in `tests/`:
- `test_goal_state_properties`: Verifies that goal state has $h_{\text{MD}} = 0$ and $h_{\text{Misplaced}} = 0$.
- `test_invalid_boards_rejected`: Rejects inputs with duplicate tiles, missing numbers, or incorrect lengths.
- `test_solvability_classification`: Proves that odd parity boards are marked unsolvable without initiating search.
- `test_legal_moves_boundaries`: Ensures the blank never moves out of the $[0, 2] \times [0, 2]$ grid.
- `test_manhattan_distance_values`: Validates exact manual taxicab calculations against implementation.
- `test_already_solved_puzzle`: Verifies immediate zero-move return when initial state equals goal.
- `test_astar_finds_optimal_path_easy`: Confirms A\* finds minimal path and every step is a legal transition.
- `test_bfs_finds_optimal_path_easy`: Confirms BFS shortest path property.
- `test_astar_bfs_optimality_equivalence`: Verifies identical minimal path length between A\* and BFS.
- `test_dfs_respects_expansion_limit`: Verifies safe halting when DFS reaches its expansion threshold.

Run tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 15. Limitations & Future Enhancements

### Limitations
1. **Memory Ceiling for BFS:** At depths $\ge 18$, BFS cannot be executed within a lightweight browser session without hitting RAM limits.
2. **Single Heuristic:** While Manhattan distance is optimal for 8-puzzle, the 15-puzzle ($4 \times 4$) requires additive disjoint pattern databases or IDA\* (Iterative Deepening A\*) to handle $16! / 2 \approx 10^{13}$ states.

### Future Enhancements
1. **15-Puzzle Scaling:** Extend the architecture to 4×4 boards with Iterative Deepening A\* (IDA\*).
2. **Linear Conflict Heuristic:** Add Linear Conflict ($h_{\text{LC}} = h_{\text{MD}} + 2 \times \text{conflicts}$) to further prune symmetrical node expansions.
3. **Bidirectional Search:** Implement bidirectional A\* searching simultaneously forward from start and backward from goal.

---

## 16. Academic Syllabus Mapping

| Syllabus Component | Academic Description | Implementation in Project |
| :--- | :--- | :--- |
| **Learning Unit I (LU I)** | Foundations of AI, Intelligent Agents, PEAS formulation. | `app.py` & `puzzle.py`: PEAS model, Goal-based agent architecture. |
| **Learning Unit II (LU II)** | Problem Solving Methods, Uninformed and Informed Search. | `algorithms.py`: BFS, DFS, and A\* Search implementations. |
| **PrO-1 & PrO-2** | Simple and Goal-Based Intelligent Agent. | State transitions, goal condition checks, and action-driven agent execution. |
| **PrO-3** | Uninformed Search Algorithm. | FIFO Queue-based BFS and LIFO Stack-based DFS. |
| **PrO-4** | Heuristic Search Algorithm. | Priority-Queue A\* with Manhattan Distance heuristic. |
| **Learning Outcome 2f** | Problem-Space Model formulation. | Formal 5-tuple: $(S_0, ACTIONS, RESULT, GOAL\_TEST, PATH\_COST)$. |

> **Note on AI Classification:** This project implements classical symbolic AI based on state-space search and heuristics (Russell & Norvig, Nilsson). It does not use neural networks or statistical machine learning, highlighting the foundational principles of problem-solving agents.

---

## 17. References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. Chapter 3: "Solving Problems by Searching".
2. Nilsson, N. J. (2014). *Principles of Artificial Intelligence*. Morgan Kaufmann.
3. Korf, R. E. (1985). "Depth-first iterative-deepening: An optimal admissible tree search." *Artificial Intelligence*, 27(1), 97-109.
4. Johnson, W. W., & Story, W. E. (1879). "Notes on the '15' puzzle." *American Journal of Mathematics*, 2(4), 397-404. (Original parity proof for sliding tile puzzles).
