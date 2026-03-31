# Streamlit Interactive UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app with three pages — Configure (2D click-to-place ships, weapon/hull config, hyperparams), Training (real-time episode animation + reward curves), and Replay (checkpoint playback) — backed by a subprocess training model.

**Architecture:** Training runs as a `subprocess.Popen` spawned by Streamlit; it writes `outputs/live_state.json` atomically each episode. Streamlit polls that file every 0.5 s via `st.fragment(run_every=0.5)`. Replay uses a separate `replay_rollout.py` subprocess that dumps a full deterministic trajectory to `outputs/replay_trajectory.npy`.

**Tech Stack:** Python 3.10+, Streamlit 1.36+, Plotly 5.18+, PyTorch (existing), NumPy (existing), PyYAML (existing)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add streamlit, plotly optional dep group `ui` |
| Create | `app.py` | Streamlit entry point — `st.navigation` routing |
| Create | `pages/__init__.py` | Empty |
| Create | `pages/configure.py` | Tab 1: battlefield placement, Tab 2: ship/weapon config, Tab 3: hyperparams + launch |
| Create | `pages/training.py` | Live animation fragment + metrics panel + stop button |
| Create | `pages/replay.py` | Checkpoint selector, rollout trigger, playback controls |
| Create | `src/naval_rl/ui_bridge.py` | `write_live_state()`, `read_live_state()`, schema constants |
| Modify | `scripts/train.py` | JSON config loading + `write_live_state()` call each episode |
| Create | `scripts/replay_rollout.py` | Deterministic rollout → `outputs/replay_trajectory.npy` |
| Create | `tests/test_ui_bridge.py` | Unit tests for bridge read/write |
| Create | `tests/test_train_json.py` | Test JSON config loading in train.py |

---

## Task 1: Add UI dependencies and app entry point

**Files:**
- Modify: `pyproject.toml`
- Create: `app.py`
- Create: `pages/__init__.py`

- [ ] **Step 1: Add `ui` optional dependency group to `pyproject.toml`**

Open `pyproject.toml` and add after the `viz` group:

```toml
ui = [
  "streamlit>=1.36",
  "plotly>=5.18",
]
```

- [ ] **Step 2: Install the new group**

```bash
pip install -e ".[ui]"
```

Expected: installs streamlit and plotly without errors.

- [ ] **Step 3: Create `pages/__init__.py`**

```python
```

(empty file — makes `pages/` a package so Streamlit can import from it)

- [ ] **Step 4: Create `app.py`**

```python
"""app.py — Streamlit entry point for the Naval RL interactive UI."""
import streamlit as st

st.set_page_config(
    page_title="Naval RL",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure = st.Page("pages/configure.py", title="Configure", icon="⚙️", default=True)
training  = st.Page("pages/training.py",  title="Training",  icon="📡")
replay    = st.Page("pages/replay.py",    title="Replay",    icon="▶️")

pg = st.navigation([configure, training, replay])
pg.run()
```

- [ ] **Step 5: Smoke-test the app launches**

```bash
streamlit run app.py
```

Expected: browser opens, three pages appear in sidebar, each shows a blank page (they don't exist yet — errors are fine at this stage, we just confirm routing works).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app.py pages/__init__.py
git commit -m "feat: add streamlit UI scaffold and ui dependency group"
```

---

## Task 2: `ui_bridge.py` — shared state I/O

**Files:**
- Create: `src/naval_rl/ui_bridge.py`
- Create: `tests/test_ui_bridge.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ui_bridge.py`:

```python
"""Tests for ui_bridge read/write roundtrip."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from naval_rl.ui_bridge import write_live_state, read_live_state, LIVE_STATE_SCHEMA


SAMPLE_STATE = {
    "episode": 5,
    "step": 100,
    "ships": [
        {"fleet": "alice", "name": "Alice-1", "x": 10000.0, "y": -5000.0,
         "heading": 1.57, "alive": True},
        {"fleet": "bob",   "name": "Bob-1",   "x": -20000.0, "y": 30000.0,
         "heading": 0.0,  "alive": True},
    ],
    "missiles": [{"x": 5000.0, "y": 0.0}],
    "rewards": {"alice": 12.4, "bob": -3.1},
    "history": [
        {"episode": 1, "r_alice": 5.2, "r_bob": 2.1},
        {"episode": 2, "r_alice": 7.0, "r_bob": 1.5},
    ],
}


def test_write_creates_file(tmp_path):
    path = tmp_path / "live_state.json"
    write_live_state(SAMPLE_STATE, path=path)
    assert path.exists()


def test_read_roundtrip(tmp_path):
    path = tmp_path / "live_state.json"
    write_live_state(SAMPLE_STATE, path=path)
    result = read_live_state(path=path)
    assert result["episode"] == 5
    assert result["step"] == 100
    assert len(result["ships"]) == 2
    assert result["ships"][0]["fleet"] == "alice"
    assert result["rewards"]["alice"] == pytest.approx(12.4)
    assert len(result["history"]) == 2


def test_write_is_atomic(tmp_path):
    """File should not be partially written (no tmp file left behind)."""
    path = tmp_path / "live_state.json"
    write_live_state(SAMPLE_STATE, path=path)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "live_state.json"


def test_read_missing_returns_none(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert read_live_state(path=path) is None


def test_schema_keys_present():
    required = {"episode", "step", "ships", "missiles", "rewards", "history"}
    assert required == set(LIVE_STATE_SCHEMA)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ui_bridge.py -v
```

Expected: `ModuleNotFoundError: No module named 'naval_rl.ui_bridge'`

- [ ] **Step 3: Implement `src/naval_rl/ui_bridge.py`**

```python
"""
ui_bridge.py — Shared state I/O between training subprocess and Streamlit UI.

Training calls write_live_state() each episode.
Streamlit calls read_live_state() every 0.5 s to animate the battlefield.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default path — both trainer and UI use this constant.
_DEFAULT_PATH = Path("outputs/live_state.json")

# Schema key reference (used in tests and validation).
LIVE_STATE_SCHEMA = {
    "episode": int,
    "step": int,
    "ships": list,       # [{"fleet", "name", "x", "y", "heading", "alive"}]
    "missiles": list,    # [{"x", "y"}]
    "rewards": dict,     # {"alice": float, "bob": float}
    "history": list,     # [{"episode": int, "r_alice": float, "r_bob": float}]
}


def write_live_state(
    state: Dict[str, Any],
    path: Path = _DEFAULT_PATH,
) -> None:
    """
    Atomically write live training state to *path*.

    Uses write-to-tmp + os.replace to prevent partial reads by Streamlit.
    No-op if the parent directory does not exist (allows import in non-UI mode).
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)
    except Exception:
        pass  # Never crash the training loop due to UI I/O


def read_live_state(
    path: Path = _DEFAULT_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Read live state from *path*. Returns None if the file does not exist
    or cannot be parsed (e.g., mid-write race condition).
    """
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/test_ui_bridge.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naval_rl/ui_bridge.py tests/test_ui_bridge.py
git commit -m "feat: add ui_bridge for atomic live-state sharing between trainer and UI"
```

---

## Task 3: Modify `train.py` — JSON config support + live state writing

**Files:**
- Modify: `scripts/train.py`
- Create: `tests/test_train_json.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_train_json.py`:

```python
"""Tests for JSON config loading in train.py."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train import load_config


def test_load_yaml(tmp_path):
    cfg = {"run_name": "test", "fleet_alice": [], "fleet_bob": []}
    p = tmp_path / "config.yaml"
    import yaml
    p.write_text(yaml.dump(cfg))
    result = load_config(str(p))
    assert result["run_name"] == "test"


def test_load_json(tmp_path):
    cfg = {"run_name": "ui_run", "fleet_alice": [], "fleet_bob": []}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    result = load_config(str(p))
    assert result["run_name"] == "ui_run"


def test_load_unknown_extension_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[foo]")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(str(p))
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_train_json.py -v
```

Expected: `test_load_json` FAILS, `test_load_unknown_extension_raises` FAILS.

- [ ] **Step 3: Update `load_config` in `scripts/train.py`**

Replace the existing `load_config` function (lines 41–43):

```python
def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        with open(p) as f:
            return yaml.safe_load(f)
    elif p.suffix == ".json":
        import json as _json
        with open(p) as f:
            return _json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
```

- [ ] **Step 4: Add `write_live_state` call in the training loop**

Add the import near the top of `scripts/train.py` (after the existing imports):

```python
try:
    from naval_rl.ui_bridge import write_live_state as _write_live_state
    _UI_BRIDGE_AVAILABLE = True
except ImportError:
    _UI_BRIDGE_AVAILABLE = False
```

In the `train()` function, after the `if ep % 50 == 0:` logging block and before the `# --- Greedy rollout ---` block, add:

```python
        # --- Live UI state ---
        if _UI_BRIDGE_AVAILABLE:
            ship_states = []
            for s in fleet_alice:
                ship_states.append({
                    "fleet": "alice", "name": s.name,
                    "x": float(s.x), "y": float(s.y),
                    "heading": float(s.course), "alive": s.alive,
                })
            for s in fleet_bob:
                ship_states.append({
                    "fleet": "bob", "name": s.name,
                    "x": float(s.x), "y": float(s.y),
                    "heading": float(s.course), "alive": s.alive,
                })
            missiles = []
            for s in fleet_alice + fleet_bob:
                for shot in s.last_shots:
                    sx, sy = shot["source"]
                    tx, ty = shot["target"]
                    missiles.append({"x": (sx + tx) / 2, "y": (sy + ty) / 2})

            live_state = {
                "episode": ep,
                "step": int(global_step),
                "ships": ship_states,
                "missiles": missiles,
                "rewards": {
                    "alice": float(ep_reward[0]),
                    "bob":   float(ep_reward[1]),
                },
                "history": [
                    {"episode": h["episode"], "r_alice": h["reward_alice"], "r_bob": h["reward_bob"]}
                    for h in episode_history
                ],
            }
            _write_live_state(live_state)
```

Add `episode_history: List[Dict] = []` just before `global_step = 0` in `train()`, and append to it at the end of each episode:

```python
        episode_history.append({
            "episode": ep,
            "reward_alice": float(ep_reward[0]),
            "reward_bob":   float(ep_reward[1]),
        })
```

- [ ] **Step 5: Run tests — all pass**

```bash
pytest tests/test_train_json.py tests/test_ui_bridge.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Smoke-test with existing YAML config**

```bash
python scripts/train.py --config configs/simple_attraction.yaml &
sleep 5
cat outputs/live_state.json | python -m json.tool | head -30
kill %1
```

Expected: valid JSON with `episode`, `step`, `ships`, `rewards`, `history` keys.

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py tests/test_train_json.py
git commit -m "feat: train.py supports JSON config and writes live_state.json each episode"
```

---

## Task 4: `replay_rollout.py` — deterministic rollout to numpy trajectory

**Files:**
- Create: `scripts/replay_rollout.py`

- [ ] **Step 1: Create `scripts/replay_rollout.py`**

```python
#!/usr/bin/env python3
"""
replay_rollout.py — Run one deterministic episode with loaded checkpoints.

Dumps full trajectory to outputs/replay_trajectory.npy for the UI to play back.

Usage:
    python scripts/replay_rollout.py \
        --config outputs/live_run_config.json \
        --alice outputs/my_run/alice_ep500.pt \
        --bob   outputs/my_run/bob_ep500.pt \
        --out   outputs/replay_trajectory.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from naval_rl.agents.td3 import TD3Agent
from naval_rl.envs.entities import ADMeasure, Ship, Weapon
from naval_rl.envs.naval_env import NavalEnv
from train import load_config, build_fleet


def run_rollout(cfg_path: str, alice_ckpt: str, bob_ckpt: str, out_path: str) -> None:
    cfg = load_config(cfg_path)

    fleet_alice = build_fleet(cfg["fleet_alice"])
    fleet_bob   = build_fleet(cfg["fleet_bob"])
    env = NavalEnv(
        fleet_alice = fleet_alice,
        fleet_bob   = fleet_bob,
        cfg_alice   = cfg["reward_alice"],
        cfg_bob     = cfg["reward_bob"],
        max_steps   = cfg["training"]["max_steps_per_episode"],
        grid_half   = cfg.get("grid_half", 100_000.0),
    )

    state_dim = env.observation_space.shape[0]
    agent_cfg = cfg["agent"]

    alice = TD3Agent(
        state_dim  = state_dim,
        action_dim = 4 * len(fleet_alice),
        noise_cfg  = agent_cfg["noise_alice"],
        lr_actor   = agent_cfg["lr_actor"],
        lr_critic  = agent_cfg["lr_critic"],
        gamma      = agent_cfg["gamma"],
        tau        = agent_cfg["tau"],
        hidden     = agent_cfg.get("hidden", 256),
        device     = cfg.get("device", "cpu"),
    )
    bob = TD3Agent(
        state_dim  = state_dim,
        action_dim = 4 * len(fleet_bob),
        noise_cfg  = agent_cfg["noise_bob"],
        lr_actor   = agent_cfg["lr_actor"],
        lr_critic  = agent_cfg["lr_critic"],
        gamma      = agent_cfg["gamma"],
        tau        = agent_cfg["tau"],
        hidden     = agent_cfg.get("hidden", 256),
        device     = cfg.get("device", "cpu"),
    )
    alice.load(alice_ckpt)
    bob.load(bob_ckpt)

    obs, _ = env.reset()
    done = False
    frames = []

    while not done:
        ship_states = []
        for s in fleet_alice:
            ship_states.append({
                "fleet": "alice", "name": s.name,
                "x": float(s.x), "y": float(s.y),
                "heading": float(s.course), "alive": s.alive,
            })
        for s in fleet_bob:
            ship_states.append({
                "fleet": "bob", "name": s.name,
                "x": float(s.x), "y": float(s.y),
                "heading": float(s.course), "alive": s.alive,
            })
        missiles = []
        for s in fleet_alice + fleet_bob:
            for shot in s.last_shots:
                sx, sy = shot["source"]
                tx, ty = shot["target"]
                missiles.append({"x": (sx + tx) / 2.0, "y": (sy + ty) / 2.0})

        frames.append({"ships": ship_states, "missiles": missiles})

        a_alice = alice.select_action_deterministic(obs)
        a_bob   = bob.select_action_deterministic(obs)
        obs, _, terminated, truncated, _ = env.step(np.concatenate([a_alice, a_bob]))
        done = terminated or truncated

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(frames, dtype=object), allow_pickle=True)
    print(f"Saved {len(frames)}-step trajectory → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--alice",   required=True)
    parser.add_argument("--bob",     required=True)
    parser.add_argument("--out",     default="outputs/replay_trajectory.npy")
    args = parser.parse_args()
    run_rollout(args.config, args.alice, args.bob, args.out)
```

- [ ] **Step 2: Smoke-test with existing config and checkpoints (if available)**

```bash
# Only run if checkpoints exist from a previous training run
ls outputs/simple_attraction/alice_final.pt 2>/dev/null && \
python scripts/replay_rollout.py \
    --config configs/simple_attraction.yaml \
    --alice outputs/simple_attraction/alice_final.pt \
    --bob   outputs/simple_attraction/bob_final.pt \
    --out   /tmp/test_traj.npy && \
python -c "import numpy as np; t=np.load('/tmp/test_traj.npy', allow_pickle=True); print(f'Frames: {len(t)}, keys: {list(t[0].keys())}')"
```

Expected output (if checkpoints exist):
```
Saved N-step trajectory → /tmp/test_traj.npy
Frames: N, keys: ['ships', 'missiles']
```

- [ ] **Step 3: Commit**

```bash
git add scripts/replay_rollout.py
git commit -m "feat: add replay_rollout.py for deterministic checkpoint playback"
```

---

## Task 5: Configure page — Tab 1: Battlefield (2D click-to-place)

**Files:**
- Create: `pages/configure.py` (Tab 1 only; Tabs 2 and 3 added in Tasks 6–7)

The 200×200 km grid maps to coordinates −100 to +100 km on each axis. Ships are stored in km in session state and converted to metres when building `run_config.json`.

Click-to-place uses a 41×41 invisible scatter trace (−100 to +100 km in 5 km steps, 1681 points) as a click-target layer. Streamlit's `st.plotly_chart(on_select="rerun")` captures the nearest point's x/y when a user clicks anywhere on the chart.

- [ ] **Step 1: Create `pages/configure.py` with Tab 1**

```python
"""pages/configure.py — Scenario configuration: battlefield, ships, hyperparams."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "alice_ships": [],   # list of {"x_km": float, "y_km": float, "heading_deg": float}
        "bob_ships":   [],
        "placing_fleet": "alice",
        "training_pid": None,
        "run_config_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Battlefield chart
# ---------------------------------------------------------------------------

_GRID_KM = np.linspace(-100, 100, 41)   # 5 km steps


def _make_battlefield_fig(
    alice_ships: List[Dict],
    bob_ships: List[Dict],
    selected_fleet: str,
) -> go.Figure:
    fig = go.Figure()

    # Invisible click-target grid (41x41 = 1681 points)
    gx, gy = np.meshgrid(_GRID_KM, _GRID_KM)
    fig.add_trace(go.Scatter(
        x=gx.ravel(), y=gy.ravel(),
        mode="markers",
        marker=dict(size=18, opacity=0.01, color="white"),
        name="grid",
        hovertemplate="(%{x:.0f} km, %{y:.0f} km)<extra></extra>",
        showlegend=False,
    ))

    # Alice ships
    if alice_ships:
        fig.add_trace(go.Scatter(
            x=[s["x_km"] for s in alice_ships],
            y=[s["y_km"] for s in alice_ships],
            mode="markers+text",
            marker=dict(size=14, color="royalblue", symbol="circle"),
            text=[f"A{i+1}" for i in range(len(alice_ships))],
            textposition="top center",
            name="Alice",
            showlegend=True,
        ))

    # Bob ships
    if bob_ships:
        fig.add_trace(go.Scatter(
            x=[s["x_km"] for s in bob_ships],
            y=[s["y_km"] for s in bob_ships],
            mode="markers+text",
            marker=dict(size=14, color="crimson", symbol="circle"),
            text=[f"B{i+1}" for i in range(len(bob_ships))],
            textposition="top center",
            name="Bob",
            showlegend=True,
        ))

    fig.update_layout(
        xaxis=dict(range=[-105, 105], title="km (East)", showgrid=True, dtick=20),
        yaxis=dict(range=[-105, 105], title="km (North)", showgrid=True, dtick=20,
                   scaleanchor="x", scaleratio=1),
        height=550,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.05),
        title=f"Click to place {'Alice' if selected_fleet == 'alice' else 'Bob'} ship",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    st.title("Configure Scenario")
    tab1, tab2, tab3 = st.tabs(["Battlefield", "Ships & Weapons", "Hyperparameters"])

    # -----------------------------------------------------------------------
    # TAB 1 — Battlefield
    # -----------------------------------------------------------------------
    with tab1:
        col_left, col_right = st.columns([3, 1])

        with col_right:
            st.session_state["placing_fleet"] = st.radio(
                "Placing ships for:",
                options=["alice", "bob"],
                format_func=lambda x: "Alice (blue)" if x == "alice" else "Bob (red)",
            )
            st.markdown("---")
            st.markdown(
                f"**Alice:** {len(st.session_state['alice_ships'])} ships  \n"
                f"**Bob:** {len(st.session_state['bob_ships'])} ships"
            )
            if st.button("Clear Alice ships"):
                st.session_state["alice_ships"] = []
                st.rerun()
            if st.button("Clear Bob ships"):
                st.session_state["bob_ships"] = []
                st.rerun()

        with col_left:
            fig = _make_battlefield_fig(
                st.session_state["alice_ships"],
                st.session_state["bob_ships"],
                st.session_state["placing_fleet"],
            )
            event = st.plotly_chart(
                fig,
                on_select="rerun",
                selection_mode="points",
                use_container_width=True,
                key="battlefield_chart",
            )

            # Handle click → place ship
            pts = event.selection.points if event.selection else []
            if pts:
                pt = pts[-1]
                new_ship = {"x_km": round(pt["x"], 1), "y_km": round(pt["y"], 1), "heading_deg": 0.0}
                fleet_key = f"{st.session_state['placing_fleet']}_ships"
                # Avoid duplicate placements on the same point
                existing = [(s["x_km"], s["y_km"]) for s in st.session_state[fleet_key]]
                if (new_ship["x_km"], new_ship["y_km"]) not in existing:
                    st.session_state[fleet_key].append(new_ship)
                    st.rerun()

        # Heading sliders for each placed ship
        if st.session_state["alice_ships"] or st.session_state["bob_ships"]:
            st.markdown("### Adjust initial headings")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Alice ships**")
                for i, ship in enumerate(st.session_state["alice_ships"]):
                    ship["heading_deg"] = st.slider(
                        f"Alice {i+1} heading (°)", 0, 359, int(ship["heading_deg"]),
                        key=f"alice_hdg_{i}"
                    )
            with cols[1]:
                st.markdown("**Bob ships**")
                for i, ship in enumerate(st.session_state["bob_ships"]):
                    ship["heading_deg"] = st.slider(
                        f"Bob {i+1} heading (°)", 0, 359, int(ship["heading_deg"]),
                        key=f"bob_hdg_{i}"
                    )


main()
```

- [ ] **Step 2: Smoke-test Tab 1**

```bash
streamlit run app.py
```

Navigate to Configure → Battlefield tab. Click on the chart — ships should appear as blue (Alice) or red (Bob) dots. Verify "Clear" buttons remove ships.

- [ ] **Step 3: Commit**

```bash
git add pages/configure.py
git commit -m "feat: configure page Tab 1 — 2D click-to-place ship battlefield"
```

---

## Task 6: Configure page — Tab 2: Ship & Weapon config

**Files:**
- Modify: `pages/configure.py` (add Tab 2 content inside `main()`)

- [ ] **Step 1: Add session state defaults for ship configs**

In `_init_state()`, add:

```python
"alice_ship_configs": {},   # keyed by ship index: {"max_speed", "health", "weapons", "ad_measures"}
"bob_ship_configs":   {},
```

- [ ] **Step 2: Add Tab 2 inside `main()`, after Tab 1's `with tab1:` block**

```python
    # -----------------------------------------------------------------------
    # TAB 2 — Ships & Weapons
    # -----------------------------------------------------------------------
    with tab2:
        if not st.session_state["alice_ships"] and not st.session_state["bob_ships"]:
            st.info("Place ships on the Battlefield tab first.")
        else:
            for fleet_key, label, color in [
                ("alice_ships", "Alice", "blue"),
                ("bob_ships",   "Bob",   "red"),
            ]:
                ships = st.session_state[fleet_key]
                cfg_key = f"{fleet_key[:-6]}_ship_configs"   # "alice_ship_configs" / "bob_ship_configs"
                if not ships:
                    continue
                st.markdown(f"### :{color}[{label} Fleet]")
                for i, _ in enumerate(ships):
                    cfg = st.session_state[cfg_key].setdefault(i, {
                        "max_speed": 20.0,
                        "health": 3,
                        "weapons": [{"name": "SSM-HARPOON", "stockpile": 8,
                                     "range_km": 50.0, "cooldown": 50,
                                     "p_hit": 0.8, "max_salvo_size": 4}],
                        "ad_measures": [{"name": "SAM-ESSM", "stockpile": 6,
                                         "range_km": 60.0, "cooldown": 5}],
                    })
                    with st.expander(f"{label} Ship {i+1}", expanded=(i == 0)):
                        c1, c2 = st.columns(2)
                        with c1:
                            cfg["max_speed"] = st.number_input(
                                "Max speed (knots)", 1.0, 60.0, float(cfg["max_speed"]),
                                key=f"{fleet_key}_{i}_speed"
                            )
                        with c2:
                            cfg["health"] = st.number_input(
                                "Hull points", 1, 20, int(cfg["health"]),
                                key=f"{fleet_key}_{i}_health"
                            )
                        st.markdown("**Weapon**")
                        w = cfg["weapons"][0]
                        wc1, wc2, wc3 = st.columns(3)
                        with wc1:
                            w["name"] = st.selectbox(
                                "Type", ["SSM-HARPOON", "Artillery", "Torpedo"],
                                index=["SSM-HARPOON", "Artillery", "Torpedo"].index(w["name"])
                                      if w["name"] in ["SSM-HARPOON", "Artillery", "Torpedo"] else 0,
                                key=f"{fleet_key}_{i}_wtype"
                            )
                            w["stockpile"] = st.number_input(
                                "Stockpile", 1, 50, int(w["stockpile"]),
                                key=f"{fleet_key}_{i}_wstock"
                            )
                        with wc2:
                            w["range_km"] = st.number_input(
                                "Range (km)", 1.0, 200.0, float(w["range_km"]),
                                key=f"{fleet_key}_{i}_wrange"
                            )
                            w["cooldown"] = st.number_input(
                                "Cooldown (steps)", 1, 200, int(w["cooldown"]),
                                key=f"{fleet_key}_{i}_wcool"
                            )
                        with wc3:
                            w["p_hit"] = st.slider(
                                "P(hit)", 0.0, 1.0, float(w["p_hit"]),
                                key=f"{fleet_key}_{i}_phit"
                            )
                            w["max_salvo_size"] = st.number_input(
                                "Max salvo", 1, 10, int(w["max_salvo_size"]),
                                key=f"{fleet_key}_{i}_salvo"
                            )
                        st.markdown("**Air Defence**")
                        ad = cfg["ad_measures"][0]
                        adc1, adc2 = st.columns(2)
                        with adc1:
                            ad["name"] = st.text_input(
                                "AD system name", ad["name"],
                                key=f"{fleet_key}_{i}_adname"
                            )
                            ad["stockpile"] = st.number_input(
                                "AD stockpile", 0, 30, int(ad["stockpile"]),
                                key=f"{fleet_key}_{i}_adstock"
                            )
                        with adc2:
                            ad["range_km"] = st.number_input(
                                "AD range (km)", 1.0, 200.0, float(ad["range_km"]),
                                key=f"{fleet_key}_{i}_adrange"
                            )
                            ad["cooldown"] = st.number_input(
                                "AD cooldown (steps)", 1, 50, int(ad["cooldown"]),
                                key=f"{fleet_key}_{i}_adcool"
                            )
```

- [ ] **Step 3: Smoke-test Tab 2**

Place ships on Tab 1, switch to Tab 2. Verify expanders appear per ship, values persist when switching tabs.

- [ ] **Step 4: Commit**

```bash
git add pages/configure.py
git commit -m "feat: configure page Tab 2 — per-ship hull and weapon configuration"
```

---

## Task 7: Configure page — Tab 3: Hyperparameters + Launch

**Files:**
- Modify: `pages/configure.py` (add Tab 3 + `_build_run_config()` + launch logic)
- Create: `outputs/` directory placeholder

- [ ] **Step 1: Add `_build_run_config()` helper before `main()`**

```python
import math


def _build_run_config(
    alice_ships: List[Dict],
    bob_ships: List[Dict],
    alice_cfgs: Dict,
    bob_cfgs: Dict,
    hp: Dict,
) -> Dict[str, Any]:
    """Convert UI session state into a train.py-compatible config dict."""

    def _fleet_cfg(ships, ship_configs, prefix):
        fleet = []
        for i, s in enumerate(ships):
            cfg = ship_configs.get(i, {})
            w = cfg.get("weapons", [{}])[0]
            ad = cfg.get("ad_measures", [{}])[0]
            fleet.append({
                "name": f"{prefix}-{i+1}",
                "x": s["x_km"] * 1000.0,   # km → m
                "y": s["y_km"] * 1000.0,
                "max_speed": cfg.get("max_speed", 20.0),
                "health": cfg.get("health", 3),
                "grid_size": [100000, 100000],
                "weapons": [{
                    "name": w.get("name", "SSM-HARPOON"),
                    "stockpile": w.get("stockpile", 8),
                    "range": w.get("range_km", 50.0) * 1000.0,
                    "cooldown": w.get("cooldown", 50),
                    "p_hit": w.get("p_hit", 0.8),
                    "max_salvo_size": w.get("max_salvo_size", 4),
                }],
                "ad_measures": [{
                    "name": ad.get("name", "SAM-ESSM"),
                    "stockpile": ad.get("stockpile", 6),
                    "range": ad.get("range_km", 60.0) * 1000.0,
                    "cooldown": ad.get("cooldown", 5),
                }],
            })
        return fleet

    n_alice = len(alice_ships)
    n_bob   = len(bob_ships)
    run_name = f"ui_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "run_name": run_name,
        "output_dir": "outputs",
        "device": "cpu",
        "grid_half": 100000,
        "fleet_alice": _fleet_cfg(alice_ships, alice_cfgs, "Alice"),
        "fleet_bob":   _fleet_cfg(bob_ships,   bob_cfgs,   "Bob"),
        "reward_alice": {
            "w_gravity": hp["alice_w_gravity"],
            "w_lj_sup": hp["alice_w_lj_sup"],
            "weapon_range": hp["alice_weapon_range_km"] * 1000,
            "w_lj_form": hp["alice_w_lj_form"],
            "d_form": 8000,
            "w_pred": hp["alice_w_pred"],
            "v_max": 617.0,
            "w_boundary": 1.0,
            "kill_reward": 100.0, "death_penalty": 100.0,
            "victory_bonus": 200.0, "defeat_penalty": 200.0,
            "reward_fire": 0.0, "time_penalty": -0.01, "p_hit": 0.8,
        },
        "reward_bob": {
            "w_gravity": hp["bob_w_gravity"],
            "w_lj_sup": hp["bob_w_lj_sup"],
            "weapon_range": hp["bob_weapon_range_km"] * 1000,
            "w_lj_form": hp["bob_w_lj_form"],
            "d_form": 8000,
            "w_pred": hp["bob_w_pred"],
            "v_max": 617.0,
            "w_boundary": 1.0,
            "kill_reward": 100.0, "death_penalty": 100.0,
            "victory_bonus": 200.0, "defeat_penalty": 200.0,
            "reward_fire": 0.0, "time_penalty": 0.01, "p_hit": 0.8,
        },
        "agent": {
            "lr_actor":  hp["lr_actor"],
            "lr_critic": hp["lr_critic"],
            "gamma": hp["gamma"],
            "tau": hp["tau"],
            "hidden": hp["hidden"],
            "noise_alice": {
                "kind": "decay",
                "dim": 4 * n_alice,
                "per": "episode",
                "gamma": hp["noise_decay"],
                "scale_min": 0.02,
                "base": {"kind": "gaussian", "dim": 4 * n_alice, "sigma": hp["noise_sigma"]},
            },
            "noise_bob": {
                "kind": "decay",
                "dim": 4 * n_bob,
                "per": "episode",
                "gamma": hp["noise_decay"],
                "scale_min": 0.02,
                "base": {"kind": "gaussian", "dim": 4 * n_bob, "sigma": hp["noise_sigma"]},
            },
        },
        "training": {
            "num_episodes": hp["num_episodes"],
            "max_steps_per_episode": hp["max_steps"],
            "warmup_steps": hp["warmup_steps"],
            "batch_size": hp["batch_size"],
            "save_every": hp["save_every"],
            "rollout_every": hp["rollout_every"],
        },
    }
```

- [ ] **Step 2: Add Tab 3 inside `main()`**

```python
    # -----------------------------------------------------------------------
    # TAB 3 — Hyperparameters + Launch
    # -----------------------------------------------------------------------
    with tab3:
        st.markdown("### Agent Hyperparameters")
        hc1, hc2 = st.columns(2)
        with hc1:
            lr_actor  = st.number_input("LR actor",  1e-5, 1e-2, 1e-4, format="%.5f")
            lr_critic = st.number_input("LR critic", 1e-5, 1e-2, 1e-3, format="%.5f")
            gamma     = st.slider("Gamma (discount)", 0.90, 0.999, 0.99)
            tau       = st.slider("Tau (soft update)", 0.001, 0.02, 0.005)
            hidden    = st.selectbox("Hidden units", [128, 256, 512], index=1)
        with hc2:
            num_episodes  = st.number_input("Episodes", 100, 50000, 2000, step=100)
            max_steps     = st.number_input("Max steps/episode", 100, 2000, 500, step=50)
            warmup_steps  = st.number_input("Warmup steps", 100, 10000, 2000, step=100)
            batch_size    = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
            save_every    = st.number_input("Checkpoint every N eps", 50, 5000, 500, step=50)
            rollout_every = st.number_input("Rollout snapshot every N eps", 10, 1000, 200, step=10)

        st.markdown("### Exploration Noise")
        nc1, nc2 = st.columns(2)
        with nc1:
            noise_sigma = st.slider("Noise sigma", 0.01, 0.5, 0.2)
        with nc2:
            noise_decay = st.slider("Noise decay (per episode)", 0.990, 0.9999, 0.995)

        st.markdown("### Reward Field Weights")
        rfc1, rfc2 = st.columns(2)
        with rfc1:
            st.markdown("**Alice**")
            alice_w_gravity      = st.slider("Gravity",    -2.0, 2.0, 0.0, key="aw_grav")
            alice_w_lj_sup       = st.slider("LJ Supremacy", 0.0, 3.0, 1.0, key="aw_ljs")
            alice_w_lj_form      = st.slider("LJ Formation", 0.0, 2.0, 0.5, key="aw_ljf")
            alice_w_pred         = st.slider("Predictive Intercept", 0.0, 2.0, 0.5, key="aw_pred")
            alice_weapon_range_km = st.number_input("Weapon range (km, for LJ)", 1.0, 200.0, 50.0, key="aw_wr")
        with rfc2:
            st.markdown("**Bob**")
            bob_w_gravity        = st.slider("Gravity",    -2.0, 2.0, -0.5, key="bw_grav")
            bob_w_lj_sup         = st.slider("LJ Supremacy", 0.0, 3.0, 0.0, key="bw_ljs")
            bob_w_lj_form        = st.slider("LJ Formation", 0.0, 2.0, 0.0, key="bw_ljf")
            bob_w_pred           = st.slider("Predictive Intercept", 0.0, 2.0, 0.0, key="bw_pred")
            bob_weapon_range_km  = st.number_input("Weapon range (km, for LJ)", 1.0, 200.0, 50.0, key="bw_wr")

        hp = dict(
            lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, tau=tau, hidden=hidden,
            num_episodes=num_episodes, max_steps=max_steps, warmup_steps=warmup_steps,
            batch_size=batch_size, save_every=save_every, rollout_every=rollout_every,
            noise_sigma=noise_sigma, noise_decay=noise_decay,
            alice_w_gravity=alice_w_gravity, alice_w_lj_sup=alice_w_lj_sup,
            alice_w_lj_form=alice_w_lj_form, alice_w_pred=alice_w_pred,
            alice_weapon_range_km=alice_weapon_range_km,
            bob_w_gravity=bob_w_gravity, bob_w_lj_sup=bob_w_lj_sup,
            bob_w_lj_form=bob_w_lj_form, bob_w_pred=bob_w_pred,
            bob_weapon_range_km=bob_weapon_range_km,
        )

        st.markdown("---")

        can_launch = (
            len(st.session_state["alice_ships"]) >= 1
            and len(st.session_state["bob_ships"]) >= 1
            and st.session_state["training_pid"] is None
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Export config as YAML", disabled=not can_launch):
                run_cfg = _build_run_config(
                    st.session_state["alice_ships"], st.session_state["bob_ships"],
                    st.session_state["alice_ship_configs"], st.session_state["bob_ship_configs"],
                    hp,
                )
                import yaml
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                yaml_path = Path(f"configs/custom_{ts}.yaml")
                yaml_path.parent.mkdir(exist_ok=True)
                yaml_path.write_text(yaml.dump(run_cfg, default_flow_style=False))
                st.success(f"Saved to {yaml_path}")

        with col_btn2:
            if not can_launch and st.session_state["training_pid"] is not None:
                st.warning("Training already running — go to the Training page.")
            elif not can_launch:
                st.button("Start Training", disabled=True,
                          help="Place at least 1 Alice ship and 1 Bob ship first.")
            else:
                if st.button("Start Training", type="primary"):
                    run_cfg = _build_run_config(
                        st.session_state["alice_ships"], st.session_state["bob_ships"],
                        st.session_state["alice_ship_configs"], st.session_state["bob_ship_configs"],
                        hp,
                    )
                    cfg_path = Path("outputs/live_run_config.json")
                    cfg_path.parent.mkdir(exist_ok=True)
                    cfg_path.write_text(json.dumps(run_cfg, indent=2))
                    st.session_state["run_config_path"] = str(cfg_path)

                    proc = subprocess.Popen(
                        [sys.executable, "scripts/train.py", "--config", str(cfg_path)],
                        cwd=str(Path(__file__).parent.parent),
                    )
                    st.session_state["training_pid"] = proc.pid
                    st.session_state["training_proc"] = proc
                    st.success(f"Training started (PID {proc.pid}). Navigate to Training page.")
```

- [ ] **Step 3: Smoke-test the full Configure flow**

```bash
streamlit run app.py
```

1. Place 1 Alice + 1 Bob ship on Battlefield tab
2. Adjust weapon config in Ships & Weapons tab
3. Go to Hyperparameters tab — verify all sliders appear
4. Click "Export config as YAML" — verify file appears in `configs/`
5. Click "Start Training" — verify PID appears, `outputs/live_run_config.json` exists
6. Kill the training process: in another terminal, `kill <PID>`

- [ ] **Step 4: Commit**

```bash
git add pages/configure.py
git commit -m "feat: configure page Tab 3 — hyperparameters, YAML export, and training launch"
```

---

## Task 8: Training page — live battlefield animation

**Files:**
- Create: `pages/training.py`

- [ ] **Step 1: Create `pages/training.py` with the live animation fragment**

```python
"""pages/training.py — Live battlefield animation and reward metrics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from naval_rl.ui_bridge import read_live_state

_GHOST_STEPS = 20   # number of trail positions to show


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arrow_endpoint(x_km: float, y_km: float, heading_rad: float, length: float = 4.0):
    """Return (ex, ey) for a heading arrow of *length* km."""
    return x_km + length * math.cos(heading_rad), y_km + length * math.sin(heading_rad)


def _build_live_fig(state: dict, ghost_history: list) -> go.Figure:
    fig = go.Figure()

    # Ghost trails
    for record in ghost_history:
        for ship in record["ships"]:
            color = "rgba(65,105,225,0.25)" if ship["fleet"] == "alice" else "rgba(220,20,60,0.25)"
            fig.add_trace(go.Scatter(
                x=[ship["x"] / 1000.0], y=[ship["y"] / 1000.0],
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False, hoverinfo="skip",
            ))

    # Current ships
    alice_added, bob_added = False, False
    for ship in state["ships"]:
        x_km = ship["x"] / 1000.0
        y_km = ship["y"] / 1000.0
        ex, ey = _arrow_endpoint(x_km, y_km, ship["heading"])
        is_alice = ship["fleet"] == "alice"
        color = "royalblue" if is_alice else "crimson"
        symbol = "circle" if ship["alive"] else "circle-open"

        show = (is_alice and not alice_added) or (not is_alice and not bob_added)
        fig.add_trace(go.Scatter(
            x=[x_km], y=[y_km],
            mode="markers",
            marker=dict(size=14, color=color, symbol=symbol),
            name=ship["fleet"].capitalize() if show else None,
            showlegend=show,
            hovertemplate=f"{ship['name']}<br>({x_km:.1f}, {y_km:.1f}) km<extra></extra>",
        ))
        if is_alice:
            alice_added = True
        else:
            bob_added = True

        if ship["alive"]:
            fig.add_trace(go.Scatter(
                x=[x_km, ex], y=[y_km, ey],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

    # Missiles
    if state["missiles"]:
        fig.add_trace(go.Scatter(
            x=[m["x"] / 1000.0 for m in state["missiles"]],
            y=[m["y"] / 1000.0 for m in state["missiles"]],
            mode="markers",
            marker=dict(size=8, color="yellow", symbol="x"),
            name="Missile",
            showlegend=True,
        ))

    fig.update_layout(
        xaxis=dict(range=[-105, 105], title="km (East)",  showgrid=True, dtick=20),
        yaxis=dict(range=[-105, 105], title="km (North)", showgrid=True, dtick=20,
                   scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.05),
        title=f"Episode {state['episode']} — Step {state['step']}",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Fragment — polls live_state.json every 0.5 s
# ---------------------------------------------------------------------------

@st.fragment(run_every=0.5)
def _live_battlefield():
    state = read_live_state()
    if state is None:
        st.info("Waiting for training to start...")
        return

    if "ghost_history" not in st.session_state:
        st.session_state["ghost_history"] = []

    history = st.session_state["ghost_history"]
    if not history or history[-1] != {"ships": [
        {k: s[k] for k in ("fleet", "x", "y", "heading")} for s in state["ships"]
    ]}:
        history.append({
            "ships": [{"fleet": s["fleet"], "x": s["x"], "y": s["y"], "heading": s["heading"]}
                      for s in state["ships"]]
        })
        if len(history) > _GHOST_STEPS:
            history.pop(0)

    fig = _build_live_fig(state, history[:-1])
    st.plotly_chart(fig, use_container_width=True, key="live_chart")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Live Training")
    _live_battlefield()


main()
```

- [ ] **Step 2: Smoke-test live animation**

Start a training run from the Configure page, then navigate to Training. Verify the chart updates approximately every 0.5 s showing ships moving. Kill training after a few episodes.

- [ ] **Step 3: Commit**

```bash
git add pages/training.py
git commit -m "feat: training page live battlefield animation with ghost trail"
```

---

## Task 9: Training page — metrics panel and stop control

**Files:**
- Modify: `pages/training.py`

- [ ] **Step 1: Add metrics fragment and stop button to `main()`**

Replace the `main()` function in `pages/training.py`:

```python
@st.fragment(run_every=1.0)
def _metrics_panel():
    state = read_live_state()
    if state is None:
        return

    history = state.get("history", [])
    if not history:
        st.info("No completed episodes yet.")
        return

    episodes   = [h["episode"]  for h in history]
    r_alice    = [h["r_alice"]  for h in history]
    r_bob      = [h["r_bob"]    for h in history]

    st.markdown(f"**Episodes completed:** {len(history)}  \n"
                f"**Global step:** {state['step']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Alice reward**")
        st.line_chart({"Episode reward": r_alice}, height=200)
    with col2:
        st.markdown("**Bob reward**")
        st.line_chart({"Episode reward": r_bob}, height=200)

    alive_alice = sum(1 for s in state["ships"] if s["fleet"] == "alice" and s["alive"])
    alive_bob   = sum(1 for s in state["ships"] if s["fleet"] == "bob"   and s["alive"])
    mc1, mc2 = st.columns(2)
    mc1.metric("Alice ships alive", alive_alice)
    mc2.metric("Bob ships alive",   alive_bob)


def main() -> None:
    st.title("Live Training")

    # Stop button
    pid = st.session_state.get("training_pid")
    if pid:
        if st.button("Stop Training", type="secondary"):
            import signal
            import os
            try:
                os.kill(pid, signal.SIGTERM)
                st.session_state["training_pid"] = None
                st.session_state["training_proc"] = None
                st.success("Training stopped.")
            except ProcessLookupError:
                st.session_state["training_pid"] = None
                st.info("Training process already finished.")
    else:
        st.info("No training running. Go to Configure to start one.")

    # Check if checkpoints exist for navigation hint
    ckpts = list(Path("outputs").glob("*/alice_*.pt"))
    if ckpts:
        st.success(f"{len(ckpts)} checkpoint(s) available — navigate to Replay when ready.")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        _live_battlefield()
    with col_right:
        _metrics_panel()


main()
```

- [ ] **Step 2: Smoke-test full training page**

Launch training, navigate to Training page. Verify:
- Battlefield animates on the left
- Reward charts update on the right
- "Stop Training" button terminates the subprocess and the process is no longer running

- [ ] **Step 3: Commit**

```bash
git add pages/training.py
git commit -m "feat: training page metrics panel, stop button, and checkpoint indicator"
```

---

## Task 10: Replay page — checkpoint playback

**Files:**
- Create: `pages/replay.py`

- [ ] **Step 1: Create `pages/replay.py`**

```python
"""pages/replay.py — Load a checkpoint, run a deterministic rollout, and play it back."""
from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_GHOST_STEPS = 20


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint_pairs() -> dict:
    """Return {tag: (alice_path, bob_path, config_path)} for all valid checkpoint pairs."""
    pairs = {}
    outputs = Path("outputs")
    if not outputs.exists():
        return pairs
    for alice_pt in sorted(outputs.glob("*/alice_*.pt")):
        tag = alice_pt.stem.replace("alice_", "")
        run_dir = alice_pt.parent
        bob_pt = run_dir / f"bob_{tag}.pt"
        # Look for the config used for this run
        cfg_path = Path("outputs/live_run_config.json")
        if bob_pt.exists() and cfg_path.exists():
            pairs[f"{run_dir.name} / {tag}"] = (str(alice_pt), str(bob_pt), str(cfg_path))
    return pairs


# ---------------------------------------------------------------------------
# Playback figure
# ---------------------------------------------------------------------------

def _build_replay_fig(frame: dict, ghost_frames: list, step: int, total: int) -> go.Figure:
    fig = go.Figure()

    for gf in ghost_frames:
        for ship in gf["ships"]:
            color = "rgba(65,105,225,0.2)" if ship["fleet"] == "alice" else "rgba(220,20,60,0.2)"
            fig.add_trace(go.Scatter(
                x=[ship["x"] / 1000.0], y=[ship["y"] / 1000.0],
                mode="markers",
                marker=dict(size=5, color=color),
                showlegend=False, hoverinfo="skip",
            ))

    alice_added, bob_added = False, False
    for ship in frame["ships"]:
        x_km = ship["x"] / 1000.0
        y_km = ship["y"] / 1000.0
        hdg  = ship["heading"]
        ex = x_km + 4.0 * math.cos(hdg)
        ey = y_km + 4.0 * math.sin(hdg)
        is_alice = ship["fleet"] == "alice"
        color  = "royalblue" if is_alice else "crimson"
        symbol = "circle" if ship["alive"] else "circle-open"
        show = (is_alice and not alice_added) or (not is_alice and not bob_added)
        fig.add_trace(go.Scatter(
            x=[x_km], y=[y_km], mode="markers",
            marker=dict(size=14, color=color, symbol=symbol),
            name=ship["fleet"].capitalize() if show else None,
            showlegend=show,
            hovertemplate=f"{ship['name']}<br>({x_km:.1f}, {y_km:.1f}) km<extra></extra>",
        ))
        if is_alice:
            alice_added = True
        else:
            bob_added = True
        if ship["alive"]:
            fig.add_trace(go.Scatter(
                x=[x_km, ex], y=[y_km, ey], mode="lines",
                line=dict(color=color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

    if frame["missiles"]:
        fig.add_trace(go.Scatter(
            x=[m["x"] / 1000.0 for m in frame["missiles"]],
            y=[m["y"] / 1000.0 for m in frame["missiles"]],
            mode="markers",
            marker=dict(size=8, color="yellow", symbol="x"),
            name="Missile", showlegend=True,
        ))

    fig.update_layout(
        xaxis=dict(range=[-105, 105], title="km (East)",  showgrid=True, dtick=20),
        yaxis=dict(range=[-105, 105], title="km (North)", showgrid=True, dtick=20,
                   scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.05),
        title=f"Replay — Step {step} / {total}",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Replay")

    pairs = _find_checkpoint_pairs()
    if not pairs:
        st.warning("No checkpoints found. Train a model first.")
        return

    # Checkpoint selector
    selected = st.selectbox("Select checkpoint", list(pairs.keys()))
    alice_pt, bob_pt, cfg_path = pairs[selected]

    traj_path = Path("outputs/replay_trajectory.npy")

    if st.button("Load & Run Deterministic Episode"):
        with st.spinner("Running deterministic rollout..."):
            result = subprocess.run(
                [sys.executable, "scripts/replay_rollout.py",
                 "--config", cfg_path,
                 "--alice",  alice_pt,
                 "--bob",    bob_pt,
                 "--out",    str(traj_path)],
                cwd=str(Path(__file__).parent.parent),
                capture_output=True, text=True,
            )
        if result.returncode != 0:
            st.error(f"Rollout failed:\n{result.stderr}")
            return
        st.success(f"Rollout complete. {result.stdout.strip()}")

    if not traj_path.exists():
        st.info("Click 'Load & Run' to generate a replay.")
        return

    # Load trajectory
    trajectory = np.load(traj_path, allow_pickle=True)
    total_steps = len(trajectory)

    # Session state for playback
    for k, v in [("replay_step", 0), ("replay_playing", False), ("replay_speed", 1.0)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Controls
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        if st.button("Play" if not st.session_state["replay_playing"] else "Pause"):
            st.session_state["replay_playing"] = not st.session_state["replay_playing"]
    with ctrl2:
        if st.button("Step Back") and st.session_state["replay_step"] > 0:
            st.session_state["replay_step"] -= 1
            st.session_state["replay_playing"] = False
    with ctrl3:
        if st.button("Step Fwd") and st.session_state["replay_step"] < total_steps - 1:
            st.session_state["replay_step"] += 1
            st.session_state["replay_playing"] = False
    with ctrl4:
        st.session_state["replay_speed"] = st.select_slider(
            "Speed", options=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
            value=st.session_state["replay_speed"]
        )

    step = st.slider("Step", 0, total_steps - 1, st.session_state["replay_step"], key="step_slider")
    st.session_state["replay_step"] = step

    frame = trajectory[step].item() if hasattr(trajectory[step], "item") else trajectory[step]
    ghost_start = max(0, step - _GHOST_STEPS)
    ghost_frames = [
        (trajectory[i].item() if hasattr(trajectory[i], "item") else trajectory[i])
        for i in range(ghost_start, step)
    ]

    col_chart, col_stats = st.columns([3, 1])
    with col_chart:
        fig = _build_replay_fig(frame, ghost_frames, step, total_steps)
        st.plotly_chart(fig, use_container_width=True, key="replay_chart")
    with col_stats:
        st.metric("Step", f"{step} / {total_steps}")
        alive_a = sum(1 for s in frame["ships"] if s["fleet"] == "alice" and s["alive"])
        alive_b = sum(1 for s in frame["ships"] if s["fleet"] == "bob"   and s["alive"])
        st.metric("Alice alive", alive_a)
        st.metric("Bob alive",   alive_b)
        st.metric("Missiles", len(frame.get("missiles", [])))

    # Auto-advance during playback
    if st.session_state["replay_playing"]:
        if st.session_state["replay_step"] < total_steps - 1:
            time.sleep(0.05 / st.session_state["replay_speed"])
            st.session_state["replay_step"] += 1
            st.rerun()
        else:
            st.session_state["replay_playing"] = False


main()
```

- [ ] **Step 2: Smoke-test the replay page**

1. Run a short training (e.g., 50 episodes) from Configure
2. Navigate to Replay
3. Select a checkpoint pair from the dropdown
4. Click "Load & Run" — verify the spinner appears and `outputs/replay_trajectory.npy` is created
5. Verify the battlefield shows the trajectory
6. Test Play/Pause, step forward/back, speed slider

- [ ] **Step 3: End-to-end flow test**

```bash
streamlit run app.py
```

Full flow: Configure → place ships → tweak weapon/hyperparams → Start Training → watch Training page animate → Stop Training → Replay page → load checkpoint → play back. Verify all transitions work without errors.

- [ ] **Step 4: Final commit**

```bash
git add pages/replay.py
git commit -m "feat: replay page — checkpoint selector, deterministic rollout, and step-by-step playback"
```

---

## Self-Review Notes

- **Spec coverage:** All spec requirements mapped: 2D click-to-place (Task 5), ship/weapon config (Task 6), hyperparams (Task 7), real-time animation (Task 8), reward curves + stop (Task 9), checkpoint replay with play/pause/scrub (Task 10).
- **Type consistency:** `write_live_state`/`read_live_state` use `Path` type consistently. `_build_run_config` converts km→m (×1000) wherever `x`, `y`, `range` are passed to the config.
- **Coordinate convention:** UI uses km; env uses metres. Conversion happens in `_build_run_config` (km×1000→m) and in chart builders (m/1000→km). Consistent throughout.
- **`ghost_history` in training page:** Accumulated across fragment reruns via `st.session_state`; capped at `_GHOST_STEPS` entries to avoid unbounded growth.
- **Replay auto-advance:** Uses `time.sleep` + `st.rerun()` — intentional for playback loop; the brief sleep controls frame rate.
