"""
app.py - Multi-Puzzle AI Solver (Clean Solid Colors)
===================================================
Vibrant 3-Color Theme:
1. Deep Cosmic Navy (#0b1120 / #111827) - Background & Surfaces
2. Solid Royal Blue (#2563eb / #1d4ed8) - 3x3 Puzzle Tiles with bold white numbers
3. Electric Cyan (#06b6d4 / #38bdf8) - Action buttons, highlights, and active moved tiles
"""

import sys
import os
import time
from typing import Tuple, Optional

import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle import (
    GOAL_STATE,
    is_valid_board,
    count_inversions,
    is_solvable,
    is_goal,
    manhattan_distance,
    get_legal_moves,
    generate_random_solvable_puzzle,
    parse_board,
    PRESET_PUZZLES
)
from algorithms import (
    bfs_solve,
    dfs_solve,
    astar_solve,
    SearchResult,
    compare_all_algorithms,
    ALGORITHM_FUNCTIONS
)
from water_jug import (
    solve_water_jug,
    is_water_jug_solvable,
    get_jug_successors,
    WaterJugResult
)

# Optional Plotly for charts
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# STREAMLIT PAGE CONFIG & 2-3 COLOR DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="AI Puzzle Solver",
    page_icon="🧩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    /* 1. Deep Cosmic Navy Background */
    .stApp {
        background-color: #0b1120 !important;
        background: radial-gradient(circle at 10% 15%, rgba(139, 92, 246, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.15) 0%, transparent 40%),
                    #0b1120 !important;
        color: #f8fafc !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Main Container max-width */
    .main .block-container {
        max-width: 530px !important;
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 8px;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 900;
        color: #38bdf8;
        letter-spacing: -0.02em;
        margin-bottom: 1px;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Target Goal Reference Bar for 8-Puzzle */
    .goal-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 9999px;
        padding: 4px 12px;
        margin: 6px auto 10px auto;
        width: fit-content;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .goal-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 5px;
        background: #2563eb;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.72rem;
    }
    .goal-empty {
        background: transparent;
        border: 1px dashed #64748b;
        color: transparent;
    }

    /* ==========================================================================
       GENERAL BUTTON STYLES (OUTSIDE BOARD)
       ========================================================================== */
    
    /* Default Secondary Buttons: Dark Slate with Cyan Text & Border */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%) !important;
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        height: 40px !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"] p,
    button[data-testid="baseButton-secondary"] p {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
    }
    button[kind="secondary"]:hover:not(:disabled) {
        background: #0284c7 !important;
        color: #ffffff !important;
        border-color: #7dd3fc !important;
        transform: translateY(-2px);
    }
    button[kind="secondary"]:hover:not(:disabled) p {
        color: #ffffff !important;
    }

    /* Primary Solve Button: Electric Cyan */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%) !important;
        background-color: #06b6d4 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        height: 42px !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.45) !important;
        transition: all 0.15s ease !important;
    }
    button[kind="primary"] p,
    button[data-testid="baseButton-primary"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.65) !important;
    }

    /* Manual Pouring Buttons inside Expander: Electric Blue */
    div[data-testid="stExpander"] button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        height: 38px !important;
        box-shadow: 0 2px 6px rgba(6, 182, 212, 0.35) !important;
    }
    div[data-testid="stExpander"] button p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        margin: 0 !important;
    }
    div[data-testid="stExpander"] button:hover {
        background: #0369a1 !important;
        border-color: #7dd3fc !important;
        transform: translateY(-1px) !important;
    }

    /* Step Playback Buttons */
    .step-btn button {
        background: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
        height: 34px !important;
        border-radius: 8px !important;
    }
    .step-btn button p {
        font-size: 0.82rem !important;
        color: #38bdf8 !important;
    }
    .step-btn button:hover:not(:disabled) {
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }
    .step-btn button:hover:not(:disabled) p {
        color: #ffffff !important;
    }

    /* ==========================================================================
       PUZZLE BOARD TILES - SOLID ROYAL BLUE & ELECTRIC CYAN (100% SPECIFICITY)
       ========================================================================== */
    /* Container enclosing the 3x3 board */
    div:has(#puzzle-board-marker),
    div[data-testid="stVerticalBlock"] > div:has(#puzzle-board-marker) {
        background: #111827 !important;
        border: 2px solid #1e293b !important;
        border-radius: 18px !important;
        padding: 10px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 8px !important;
    }

    /* Hide the zero-height anchor wrapper */
    div:has(#puzzle-board-marker):not(:has(button)) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Solid Royal Blue Puzzle Tiles with Crisp Bold White Text */
    div:has(#puzzle-board-marker) ~ div button,
    div:has(#puzzle-board-marker) ~ div[data-testid="stHorizontalBlock"] button,
    div:has(> * > #puzzle-board-marker) ~ div button,
    div:has(> * > #puzzle-board-marker) ~ div[data-testid="stHorizontalBlock"] button {
        background: #2563eb !important;
        background-color: #2563eb !important;
        background-image: linear-gradient(145deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 2px solid #60a5fa !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        height: 64px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.45) !important;
        padding: 0 !important;
        transition: all 0.12s ease !important;
    }
    div:has(#puzzle-board-marker) ~ div button p,
    div:has(#puzzle-board-marker) ~ div[data-testid="stHorizontalBlock"] button p,
    div:has(> * > #puzzle-board-marker) ~ div button p,
    div:has(> * > #puzzle-board-marker) ~ div[data-testid="stHorizontalBlock"] button p {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    div:has(#puzzle-board-marker) ~ div button:hover:not(:disabled),
    div:has(> * > #puzzle-board-marker) ~ div button:hover:not(:disabled) {
        background: #1d4ed8 !important;
        border-color: #93c5fd !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.6) !important;
    }

    /* Moved Active Tile: Electric Cyan with Golden Border */
    div:has(#puzzle-board-marker) ~ div button[kind="primary"],
    div:has(#puzzle-board-marker) ~ div button[data-testid="baseButton-primary"],
    div:has(> * > #puzzle-board-marker) ~ div button[kind="primary"] {
        background: #06b6d4 !important;
        background-color: #06b6d4 !important;
        background-image: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%) !important;
        border: 3px solid #facc15 !important;
        box-shadow: 0 0 18px rgba(6, 182, 212, 0.7) !important;
    }

    /* Empty Slot: Recessed Dark Cell */
    div:has(#puzzle-board-marker) ~ div button:disabled,
    div:has(> * > #puzzle-board-marker) ~ div button:disabled {
        background: #0b1120 !important;
        background-color: #0b1120 !important;
        background-image: none !important;
        border: 2px dashed #475569 !important;
        color: transparent !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.7) !important;
        cursor: default !important;
    }
    div:has(#puzzle-board-marker) ~ div button:disabled p,
    div:has(> * > #puzzle-board-marker) ~ div button:disabled p {
        color: transparent !important;
    }

    /* ==========================================================================
       WATER JUG VISUAL CYLINDER STYLES
       ========================================================================== */
    .jugs-wrapper {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 28px;
        margin: 10px auto;
        padding: 12px 14px;
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 16px;
        max-width: 360px;
    }
    .jug-column {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .jug-cylinder {
        width: 100px;
        height: 140px;
        background: rgba(15, 23, 42, 0.8);
        border: 2.5px solid #38bdf8;
        border-top: none;
        border-radius: 0 0 14px 14px;
        position: relative;
        display: flex;
        flex-direction: column-reverse;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.7), 0 0 15px rgba(56, 189, 248, 0.25);
    }
    .jug-water {
        background: linear-gradient(180deg, #38bdf8 0%, #2563eb 100%);
        width: 100%;
        transition: height 0.35s ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 2px 6px rgba(255, 255, 255, 0.4);
    }
    .jug-water-val {
        font-size: 1.2rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
    }
    .jug-caption {
        text-align: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: #38bdf8;
        margin-top: 6px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .status-info {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .status-success {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# PUZZLE MODE SELECTION (8-PUZZLE vs WATER JUG)
# ==============================================================================
puzzle_mode = st.radio(
    "Select AI Problem:",
    ["🧩 8-Puzzle (Sliding Tiles)", "🧪 Water Jug Problem"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)


# ==============================================================================
# ==============================================================================
# MODE 1: 8-PUZZLE SLIDING TILES SOLVER
# ==============================================================================
# ==============================================================================
if puzzle_mode == "🧩 8-Puzzle (Sliding Tiles)":

    # Session State
    if "board" not in st.session_state:
        st.session_state.board = PRESET_PUZZLES["Easy (4 Moves)"]["state"]
    if "initial_board" not in st.session_state:
        st.session_state.initial_board = st.session_state.board
    if "solution" not in st.session_state:
        st.session_state.solution = None
    if "step" not in st.session_state:
        st.session_state.step = 0

    def handle_tile_click(clicked_idx: int):
        board = list(st.session_state.board)
        blank_idx = board.index(0)
        r_click, c_click = divmod(clicked_idx, 3)
        r_blank, c_blank = divmod(blank_idx, 3)
        if abs(r_click - r_blank) + abs(c_click - c_blank) == 1:
            board[blank_idx], board[clicked_idx] = board[clicked_idx], board[blank_idx]
            st.session_state.board = tuple(board)
            st.session_state.solution = None
            st.session_state.step = 0

    def load_preset(preset_key: str):
        state = PRESET_PUZZLES[preset_key]["state"]
        st.session_state.board = state
        st.session_state.initial_board = state
        st.session_state.solution = None
        st.session_state.step = 0

    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧩 8-Puzzle AI Solver</div>
        <div class="app-subtitle">Click tiles to slide them, or click <strong>Solve with AI</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Top Controls
    c_diff, c_algo, c_new = st.columns([1.7, 1.5, 1.0])

    with c_diff:
        difficulty = st.selectbox(
            "Difficulty:",
            [
                "Easy (2 Moves)",
                "Easy (4 Moves)",
                "Easy (6 Moves)",
                "Medium (8 Moves)",
                "Medium (12 Moves)",
                "Hard (16 Moves)",
                "Challenge (20 Moves)",
                "Already Solved",
                "Unsolvable Demo"
            ],
            index=1,
            label_visibility="collapsed"
        )

    with c_algo:
        algo_choice = st.selectbox(
            "Algorithm:",
            ["A* (Fast & Optimal)", "BFS (Breadth-First)", "DFS (Depth-First)"],
            index=0,
            label_visibility="collapsed"
        )

    with c_new:
        if st.button("🎲 New", use_container_width=True):
            load_preset(difficulty)
            st.rerun()

    # Target Goal Bar
    st.markdown("""
    <div class="goal-bar">
        <span>🎯 Target:</span>
        <span class="goal-badge">1</span>
        <span class="goal-badge">2</span>
        <span class="goal-badge">3</span>
        <span class="goal-badge">4</span>
        <span class="goal-badge">5</span>
        <span class="goal-badge">6</span>
        <span class="goal-badge">7</span>
        <span class="goal-badge">8</span>
        <span class="goal-badge goal-empty">·</span>
    </div>
    """, unsafe_allow_html=True)

    # 3x3 Board in dedicated border wrapper
    curr_board = st.session_state.board
    sol = st.session_state.solution
    step = st.session_state.step

    active_moved_tile = None
    if sol is not None and sol.success and step > 0:
        active_moved_tile = sol.moved_tiles[step - 1]

    with st.container(border=True):
        st.markdown('<div id="puzzle-board-marker"></div>', unsafe_allow_html=True)
        for row in range(3):
            cols = st.columns(3, gap="small")
            for col in range(3):
                idx = row * 3 + col
                val = curr_board[idx]

                with cols[col]:
                    if val == 0:
                        st.button(" ", key=f"t_{idx}", disabled=True, use_container_width=True)
                    else:
                        btn_type = "primary" if (active_moved_tile is not None and val == active_moved_tile) else "secondary"
                        if st.button(f"{val}", key=f"t_{idx}", type=btn_type, use_container_width=True):
                            handle_tile_click(idx)
                            st.rerun()

    # Status
    if is_goal(curr_board):
        st.balloons()
        st.markdown("""
        <div style="text-align: center;">
            <span class="status-badge status-success">🎉 PUZZLE SOLVED! All tiles in goal positions!</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        solvable = is_solvable(curr_board)
        h_dist = manhattan_distance(curr_board)
        if not solvable:
            st.markdown("""
            <div style="text-align: center;">
                <span class="status-badge status-error">⚠️ Unsolvable Board (Odd Inversion Parity)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="status-badge status-info">Distance to Goal: ~{h_dist} moves (h = {h_dist})</span>
            </div>
            """, unsafe_allow_html=True)

    # Actions (Solve & Reset)
    act_c1, act_c2 = st.columns([1.6, 1.0], gap="small")

    with act_c1:
        solve_clicked = st.button("🚀 Solve with AI", type="primary", use_container_width=True)

    with act_c2:
        reset_clicked = st.button("↺ Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.board = st.session_state.initial_board
        st.session_state.solution = None
        st.session_state.step = 0
        st.rerun()

    if solve_clicked:
        algo_key = "A* Search (Manhattan)"
        if "BFS" in algo_choice:
            algo_key = "Breadth-First Search (BFS)"
        elif "DFS" in algo_choice:
            algo_key = "Depth-First Search (DFS)"

        with st.spinner("AI finding shortest path..."):
            result = ALGORITHM_FUNCTIONS[algo_key](st.session_state.board)
            st.session_state.solution = result
            st.session_state.step = 0
            st.rerun()

    # Playback
    if st.session_state.solution is not None:
        res = st.session_state.solution
        if res.success:
            st.markdown("---")
            total_steps = len(res.path) - 1

            st.markdown(f"""
            <div style="text-align:center; margin-bottom: 6px;">
                <span style="font-size:1rem; font-weight:800; color:#38bdf8;">
                    ✅ Solved in {res.cost} moves!
                </span>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left: 6px;">
                    ({res.execution_time_ms:.2f} ms • {res.nodes_expanded} nodes)
                </span>
            </div>
            """, unsafe_allow_html=True)

            if total_steps > 0:
                new_step = st.slider("Scrubber:", 0, total_steps, st.session_state.step, label_visibility="collapsed")
                if new_step != st.session_state.step:
                    st.session_state.step = new_step
                    st.session_state.board = res.path[new_step]
                    st.rerun()

                p1, p2, p3, p4, p5 = st.columns(5, gap="small")
                with p1:
                    if st.button("⏮ First", use_container_width=True, disabled=(st.session_state.step == 0)):
                        st.session_state.step = 0
                        st.session_state.board = res.path[0]
                        st.rerun()

                with p2:
                    if st.button("◀ Prev", use_container_width=True, disabled=(st.session_state.step == 0)):
                        st.session_state.step = max(0, st.session_state.step - 1)
                        st.session_state.board = res.path[st.session_state.step]
                        st.rerun()

                with p3:
                    if st.button("▶ Play", use_container_width=True):
                        for s in range(st.session_state.step, len(res.path)):
                            st.session_state.step = s
                            st.session_state.board = res.path[s]
                            time.sleep(0.35)
                            st.rerun()

                with p4:
                    if st.button("Next ▶", use_container_width=True, disabled=(st.session_state.step == total_steps)):
                        st.session_state.step = min(total_steps, st.session_state.step + 1)
                        st.session_state.board = res.path[st.session_state.step]
                        st.rerun()

                with p5:
                    if st.button("Last ⏭", use_container_width=True, disabled=(st.session_state.step == total_steps)):
                        st.session_state.step = total_steps
                        st.session_state.board = res.path[-1]
                        st.rerun()

                if st.session_state.step > 0:
                    action = res.actions[st.session_state.step - 1]
                    t_val = res.moved_tiles[st.session_state.step - 1]
                    st.markdown(f"""
                    <div style="text-align:center; color:#94a3b8; font-size:0.82rem; margin-top:3px;">
                        Move <strong>{st.session_state.step} of {total_steps}</strong>: 
                        Tile <span style="color:#38bdf8; font-weight:800;">{t_val}</span> moved <em>{action}</em>
                    </div>
                    """, unsafe_allow_html=True)
        elif res.status == "UNSOLVABLE":
            st.error(f"❌ {res.termination_reason}")
        else:
            st.warning(f"⚠️ {res.termination_reason}")


# ==============================================================================
# ==============================================================================
# MODE 2: WATER JUG PROBLEM SOLVER
# ==============================================================================
# ==============================================================================
else:
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧪 Water Jug AI Solver</div>
        <div class="app-subtitle">Measure the exact target volume using 2 jugs with AI search!</div>
    </div>
    """, unsafe_allow_html=True)

    # Session State for Water Jug
    if "jug_cap_a" not in st.session_state:
        st.session_state.jug_cap_a = 4
    if "jug_cap_b" not in st.session_state:
        st.session_state.jug_cap_b = 3
    if "jug_target" not in st.session_state:
        st.session_state.jug_target = 2
    if "jug_state" not in st.session_state:
        st.session_state.jug_state = (0, 0)
    if "jug_solution" not in st.session_state:
        st.session_state.jug_solution = None
    if "jug_step" not in st.session_state:
        st.session_state.jug_step = 0

    # Presets Bar
    w_p1, w_p2 = st.columns([2.5, 1.0])
    with w_p1:
        jug_preset = st.selectbox(
            "Quick Problem Presets:",
            [
                "Classic (Jug A: 4L, Jug B: 3L -> Measure 2L)",
                "Medium (Jug A: 5L, Jug B: 3L -> Measure 4L)",
                "Challenging (Jug A: 7L, Jug B: 5L -> Measure 4L)",
                "Unsolvable Demo (Jug A: 6L, Jug B: 4L -> Measure 3L)"
            ],
            index=0,
            label_visibility="collapsed"
        )
    with w_p2:
        if st.button("📥 Load", use_container_width=True):
            if "Classic" in jug_preset:
                st.session_state.jug_cap_a, st.session_state.jug_cap_b, st.session_state.jug_target = 4, 3, 2
            elif "Medium" in jug_preset:
                st.session_state.jug_cap_a, st.session_state.jug_cap_b, st.session_state.jug_target = 5, 3, 4
            elif "Challenging" in jug_preset:
                st.session_state.jug_cap_a, st.session_state.jug_cap_b, st.session_state.jug_target = 7, 5, 4
            elif "Unsolvable" in jug_preset:
                st.session_state.jug_cap_a, st.session_state.jug_cap_b, st.session_state.jug_target = 6, 4, 3
            st.session_state.jug_state = (0, 0)
            st.session_state.jug_solution = None
            st.session_state.jug_step = 0
            st.rerun()

    # Current capacities
    cap_a = st.session_state.jug_cap_a
    cap_b = st.session_state.jug_cap_b
    target = st.session_state.jug_target
    curr_a, curr_b = st.session_state.jug_state

    # Compact Visual Water Jug Cylinders
    pct_a = int((curr_a / cap_a) * 100) if cap_a > 0 else 0
    pct_b = int((curr_b / cap_b) * 100) if cap_b > 0 else 0

    jugs_html = f"""
    <div class="jugs-wrapper">
        <div class="jug-column">
            <div class="jug-cylinder">
                <div class="jug-water" style="height: {pct_a}%;">
                    <span class="jug-water-val">{curr_a}L</span>
                </div>
            </div>
            <div class="jug-caption">Jug A ({cap_a}L)</div>
        </div>
        <div class="jug-column">
            <div class="jug-cylinder">
                <div class="jug-water" style="height: {pct_b}%;">
                    <span class="jug-water-val">{curr_b}L</span>
                </div>
            </div>
            <div class="jug-caption">Jug B ({cap_b}L)</div>
        </div>
    </div>
    """
    st.markdown(jugs_html, unsafe_allow_html=True)

    # Target Goal Pill
    if curr_a == target or curr_b == target:
        st.balloons()
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="status-badge status-success">🎉 TARGET REACHED! Exact {target}L measured in jug!</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        is_solv, reason = is_water_jug_solvable(cap_a, cap_b, target)
        if not is_solv:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="status-badge status-error">⚠️ Impossible Target: {reason}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="status-badge status-info">🎯 Target: Measure exactly {target}L in either jug</span>
            </div>
            """, unsafe_allow_html=True)

    # Manual Pouring Controls inside Expander
    with st.expander("🖐️ Manual Pouring Controls (Play Yourself)", expanded=False):
        m1, m2 = st.columns(2, gap="small")
        with m1:
            if st.button(f"💧 Fill Jug A ({cap_a}L)", use_container_width=True):
                st.session_state.jug_state = (cap_a, curr_b)
                st.rerun()
            if st.button("🫗 Pour A → B", use_container_width=True):
                transfer = min(curr_a, cap_b - curr_b)
                st.session_state.jug_state = (curr_a - transfer, curr_b + transfer)
                st.rerun()
            if st.button("🚰 Empty Jug A", use_container_width=True):
                st.session_state.jug_state = (0, curr_b)
                st.rerun()

        with m2:
            if st.button(f"💧 Fill Jug B ({cap_b}L)", use_container_width=True):
                st.session_state.jug_state = (curr_a, cap_b)
                st.rerun()
            if st.button("🫗 Pour B → A", use_container_width=True):
                transfer = min(curr_b, cap_a - curr_a)
                st.session_state.jug_state = (curr_a + transfer, curr_b - transfer)
                st.rerun()
            if st.button("🚰 Empty Jug B", use_container_width=True):
                st.session_state.jug_state = (curr_a, 0)
                st.rerun()

    # Action Buttons: Solve with AI (BFS) & Reset
    j_act1, j_act2 = st.columns([1.6, 1.0], gap="small")
    with j_act1:
        j_solve_clicked = st.button("🚀 Solve with AI (BFS)", type="primary", use_container_width=True)

    with j_act2:
        j_reset_clicked = st.button("↺ Reset Jugs", use_container_width=True)

    if j_reset_clicked:
        st.session_state.jug_state = (0, 0)
        st.session_state.jug_solution = None
        st.session_state.jug_step = 0
        st.rerun()

    if j_solve_clicked:
        with st.spinner("AI finding optimal pouring steps..."):
            j_res = solve_water_jug(cap_a, cap_b, target)
            st.session_state.jug_solution = j_res
            st.session_state.jug_step = 0
            if j_res.success and len(j_res.path) > 0:
                st.session_state.jug_state = j_res.path[0]
            st.rerun()

    # Solution Playback
    if st.session_state.jug_solution is not None:
        j_res = st.session_state.jug_solution
        if j_res.success:
            st.markdown("---")
            j_total_steps = len(j_res.path) - 1

            st.markdown(f"""
            <div style="text-align:center; margin-bottom: 6px;">
                <span style="font-size:1rem; font-weight:800; color:#38bdf8;">
                    ✅ Optimal Solution: {j_res.cost} moves!
                </span>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left: 6px;">
                    ({j_res.execution_time_ms:.2f} ms • {j_res.nodes_expanded} states explored)
                </span>
            </div>
            """, unsafe_allow_html=True)

            if j_total_steps > 0:
                new_j_step = st.slider("Scrubber:", 0, j_total_steps, st.session_state.jug_step, label_visibility="collapsed")
                if new_j_step != st.session_state.jug_step:
                    st.session_state.jug_step = new_j_step
                    st.session_state.jug_state = j_res.path[new_j_step]
                    st.rerun()

                jp1, jp2, jp3, jp4, jp5 = st.columns(5, gap="small")
                with jp1:
                    if st.button("⏮ First", key="j_first", use_container_width=True, disabled=(st.session_state.jug_step == 0)):
                        st.session_state.jug_step = 0
                        st.session_state.jug_state = j_res.path[0]
                        st.rerun()

                with jp2:
                    if st.button("◀ Prev", key="j_prev", use_container_width=True, disabled=(st.session_state.jug_step == 0)):
                        st.session_state.jug_step = max(0, st.session_state.jug_step - 1)
                        st.session_state.jug_state = j_res.path[st.session_state.jug_step]
                        st.rerun()

                with jp3:
                    if st.button("▶ Play", key="j_play", use_container_width=True):
                        for s in range(st.session_state.jug_step, len(j_res.path)):
                            st.session_state.jug_step = s
                            st.session_state.jug_state = j_res.path[s]
                            time.sleep(0.4)
                            st.rerun()

                with jp4:
                    if st.button("Next ▶", key="j_next", use_container_width=True, disabled=(st.session_state.jug_step == j_total_steps)):
                        st.session_state.jug_step = min(j_total_steps, st.session_state.jug_step + 1)
                        st.session_state.jug_state = j_res.path[st.session_state.jug_step]
                        st.rerun()

                with jp5:
                    if st.button("Last ⏭", key="j_last", use_container_width=True, disabled=(st.session_state.jug_step == j_total_steps)):
                        st.session_state.jug_step = j_total_steps
                        st.session_state.jug_state = j_res.path[-1]
                        st.rerun()

                if st.session_state.jug_step > 0:
                    action_txt = j_res.actions[st.session_state.jug_step - 1]
                    ja, jb = st.session_state.jug_state
                    st.markdown(f"""
                    <div style="text-align:center; color:#94a3b8; font-size:0.82rem; margin-top:3px;">
                        Move <strong>{st.session_state.jug_step} of {j_total_steps}</strong>: 
                        <span style="color:#38bdf8; font-weight:700;">{action_txt}</span> 
                        (Jug A: <strong>{ja}L</strong>, Jug B: <strong>{jb}L</strong>)
                    </div>
                    """, unsafe_allow_html=True)
        elif j_res.status == "UNSOLVABLE":
            st.error(f"❌ {j_res.message}")


# ==============================================================================
# COLLAPSIBLE BENCHMARK & PROJECT DETAILS (FOR EXAMINERS)
# ==============================================================================
with st.expander("📊 Benchmark & Project Details (For Presentation / Viva)", expanded=False):
    st.markdown("### Search Algorithms Analysis")
    if st.button("Run 8-Puzzle Comparison (BFS vs DFS vs A*)"):
        with st.spinner("Benchmarking on current puzzle..."):
            bench = compare_all_algorithms(st.session_state.get("initial_board", (1,2,3,4,0,5,6,7,8)))
            rows = [r.to_dict() for r in bench.values()]
            st.dataframe(rows, use_container_width=True, hide_index=True)

            if HAS_PLOTLY:
                algos = list(bench.keys())
                expanded = [bench[k].nodes_expanded for k in algos]
                fig = go.Figure(data=[
                    go.Bar(x=algos, y=expanded, marker_color=['#38bdf8', '#2563eb', '#64748b'])
                ])
                fig.update_layout(
                    title="Nodes Expanded (Lower is Better)",
                    template="plotly_dark",
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                    height=240,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ---
    #### Quick Viva Reference:
    - **8-Puzzle:** State space has $181,440$ reachable states. Solvable iff inversion parity is even. A* with Manhattan distance finds optimal solution exploring >95% fewer states than BFS.
    - **Water Jug Problem:** State space is $(C_A + 1) \times (C_B + 1)$. Solvable by Bézout's identity iff $\text{Target} \le \max(C_A, C_B)$ and $\text{Target} \pmod{\gcd(C_A, C_B)} == 0$. BFS guarantees minimum pouring operations.
    """)
