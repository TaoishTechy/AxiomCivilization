#!/usr/bin/env python3
# axiom-gui.py — Axiom Civilization Simulator GUI
# Streamlit dashboard to generate, inspect, and simulate Civilization Axioms
# Compatible with axiomforge_civilization-0.1.py (Civilization Horizon edition)
#
# Features
# - One-click generation of axioms via the CLI script (or import fallback)
# - Deep inspection of each axiom's structures (mechanisms, metrics, games)
# - 12 "out-of-this-world" simulation/analysis functions operating on the JSON
# - Pareto frontier sketching, replicator dynamics, consensus robustness, etc.
# - Save/Load sessions; export CSV; quicklinks for repo hygiene
#
# Usage:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run axiom-gui.py
#
# Expected files in the same folder (or configure in sidebar):
#   - axiomforge_civilization-0.1.py
#   - data JSONs (adjectives.json, nouns.json, verbs.json, concepts.json, ...)
#
# This GUI does not depend on private APIs; it shells out to the generator and
# parses its stdout. It also works with pre-saved Civilization_Session.json.

import json
import os
import re
import subprocess
import sys
import time
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit servers
import matplotlib.pyplot as plt

import streamlit as st

APP_TITLE = "Axiom Civilization Simulator (GUI)"
GENERATOR_FILENAME_DEFAULT = "axiomforge_civilization-0.1.py"


# --------------------------- Utility / IO helpers ---------------------------

def robust_json_parse_blocks(stdout: str) -> List[Dict[str, Any]]:
    """
    Parse the stdout produced by axiomforge_civilization-0.1.py.
    The generator prints markers like:
      --- CIVILIZATION AXIOM ---
      { JSON ... }
    We split on the marker and load each JSON block.
    """
    blocks: List[Dict[str, Any]] = []
    parts = re.split(r"--+\s*CIVILIZATION AXIOM\s*--+\s*", stdout)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Find first JSON object in the part
        try:
            first_brace = part.find("{")
            last_brace = part.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                js = part[first_brace:last_brace+1]
                blocks.append(json.loads(js))
        except Exception:
            # Skip malformed block
            continue
    return blocks


def try_import_generator(module_path: Path) -> Optional[Any]:
    """
    Try importing the generator as a module to call internal builders.
    Most robust path remains the CLI, but import can be handy for future use.
    """
    try:
        sys.path.insert(0, str(module_path.parent))
        return __import__(module_path.stem.replace("-", "_"))
    except Exception:
        return None


def run_generator_cli(generator_path: Path,
                      seed: str,
                      tone: str = "civilization",
                      count: int = 12,
                      rng: Optional[int] = None,
                      max_mech: int = 3,
                      tao: bool = False,
                      civilization: bool = True,
                      math_theme: str = "auto",
                      selfref_mode: str = "meta",
                      save_to_history: bool = True) -> Tuple[List[Dict[str, Any]], str]:
    """
    Invoke the axiom generator via subprocess and capture stdout.
    Returns (list_of_axioms, raw_stdout)
    """
    cmd = [sys.executable, str(generator_path),
           "--seed", seed,
           "--tone", tone,
           "--count", str(count),
           "--max-mech", str(max_mech),
           "--math-theme", math_theme,
           "--selfref-mode", selfref_mode,
          ]
    if rng is not None:
        cmd += ["--rng", str(rng)]
    if tao:
        cmd.append("--tao")
    if civilization:
        cmd.append("--civilization")
    if save_to_history:
        cmd.append("--save-to-history")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = proc.stdout
    stderr = proc.stderr

    axioms = robust_json_parse_blocks(stdout)
    if not axioms and stderr:
        # Some generators print their JSON to stderr in edge cases
        axioms = robust_json_parse_blocks(stderr)
    return axioms, stdout + "\n" + stderr


def load_saved_session(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return []


def export_axioms_csv(axioms: List[Dict[str, Any]], path: Path) -> None:
    flat_rows = []
    for ax in axioms:
        row = {}
        for k, v in ax.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        flat_rows.append(row)
    df = pd.DataFrame(flat_rows)
    df.to_csv(path, index=False)


# --------------------------- Data Access helpers ---------------------------

def get(ax: Dict[str, Any], path: str, default=None):
    cur = ax
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def hash_seed(s: str) -> int:
    return int(md5(s.encode("utf-8")).hexdigest()[:8], 16)


# --------------------------- 12 Out-Of-This-World sims ---------------------
# All functions are pure/side-effect-free wrt the axiom JSON and use defaults
# where fields are missing to ensure compatibility.

def oow_1_replicator_dynamics(ax: Dict[str, Any], steps: int = 120) -> pd.DataFrame:
    """
    OUT-OF-THIS-WORLD #1: Replicator Dynamics across the declared strategy_space.
    Creates a symmetric payoff matrix seeded by the axiom's timestamp hash.
    Returns a DataFrame of strategy proportions over time.
    """
    es = get(ax, "evolutionary_stability", {})
    strategies = es.get("strategy_space", ["tit_for_tat", "always_cooperate", "always_defect", "pavlov"])
    n = len(strategies)
    if n < 2:
        strategies = ["A", "B"]
        n = 2

    # Seed from timestamp/seed for determinism
    ts = ax.get("timestamp", "") + ax.get("seed_concept", "")
    rng = np.random.default_rng(hash_seed(ts))

    # Random symmetric payoff matrix with positive diagonal bonus
    M = rng.normal(loc=0.0, scale=1.0, size=(n, n))
    M = (M + M.T) / 2.0
    for i in range(n):
        M[i, i] += abs(M).mean()

    x = np.ones(n) / n  # uniform initial mix
    traj = [x.copy()]
    for _ in range(steps):
        fitness = M @ x
        avg_fit = float(x @ fitness)
        x = x * np.maximum(1e-9, fitness) / max(1e-9, avg_fit)
        x = x / x.sum()
        traj.append(x.copy())
    df = pd.DataFrame(traj, columns=strategies)
    df["t"] = range(len(traj))
    return df


def oow_2_attention_gauge(ax: Dict[str, Any]) -> float:
    """
    OUT-OF-THIS-WORLD #2: Attention Gauge Flux Score.
    Combines metrics (density, elegance) and value gradients to estimate an
    'attention current' magnitude in [0, 1.5].
    """
    metrics = ax.get("metrics", {})
    density = float(metrics.get("density", 10.0))
    elegance = float(metrics.get("elegance", 95.0))
    vl = ax.get("value_landscape", {}).get("value_gradients", {})
    vg = np.array(list(vl.values())) if vl else np.array([0.0])
    norm_grad = float(np.linalg.norm(vg)) if vg.size else 0.0
    score = 0.01 * density + 0.005 * (100 - elegance) + 0.2 * min(1.0, norm_grad)
    return float(min(1.5, max(0.0, score)))


def oow_3_trust_emergence_index(ax: Dict[str, Any]) -> float:
    """
    OUT-OF-THIS-WORLD #3: Trust Emergence Index in [0, 1].
    Uses signaling_equilibrium, verification_mechanisms, and information_asymmetry.
    """
    eq = ax.get("signaling_equilibrium", "hybrid_equilibrium")
    vm = set(ax.get("verification_mechanisms", []))
    ia = get(ax, "strategic_information.information_asymmetry", 0.5)

    base = {"separating_equilibrium": 0.7, "hybrid_equilibrium": 0.55, "pooling_equilibrium": 0.4}.get(eq, 0.5)
    verif_bonus = 0.1 * len(vm.intersection({"zero_knowledge_proofs", "trusted_execution", "social_verification", "economic_bonding"}))
    asym_penalty = 0.4 * max(0.0, float(ia) - 0.5)
    return float(min(1.0, max(0.0, base + verif_bonus - asym_penalty)))


def oow_4_coordination_scaling_curve(ax: Dict[str, Any], max_scale: Optional[int] = None) -> pd.DataFrame:
    """
    OUT-OF-THIS-WORLD #4: Coordination scaling projection.
    Extrapolates coordination_efficiency across log-scale sizes around the axiom's scale.
    """
    cc = ax.get("civilization_coordination", {})
    base_scale = float(cc.get("scale", 1e6))
    base_eff = float(cc.get("coordination_efficiency", 0.7))
    if base_scale <= 0:
        base_scale = 10.0

    scales = np.logspace(np.log10(base_scale/100), np.log10(base_scale*100), 40)
    # Simple elasticity model: efficiency degrades slightly with scale unless algorithmic_mediation
    mech = cc.get("coordination_mechanism", "social_norms")
    elasticity = {"algorithmic_mediation": -0.02, "price_system": -0.06, "voting_mechanism": -0.08, "social_norms": -0.1}.get(mech, -0.07)
    eff = base_eff * (scales / base_scale) ** elasticity
    df = pd.DataFrame({"scale": scales, "efficiency": np.clip(eff, 0.1, 0.99)})
    return df


def oow_5_consensus_robustness(ax: Dict[str, Any]) -> float:
    """
    OUT-OF-THIS-WORLD #5: Consensus Robustness Score in [0, 1].
    Derived from protocol, fault_tolerance, n_nodes, latency and attack vectors.
    """
    dc = ax.get("distributed_consensus", {})
    ft = int(dc.get("fault_tolerance", 1))
    n = int(dc.get("n_nodes", 10))
    latency = dc.get("latency", "seconds")
    attacks = set(ax.get("attack_vectors", []))

    proto_bonus = {"practical_byzantine_fault_tolerance": 0.2, "federated_byzantine_agreement": 0.18,
                   "hashgraph_consensus": 0.16, "proof_of_stake": 0.12, "delegated_proof_of_stake": 0.1,
                   "proof_of_work": 0.08}.get(dc.get("protocol", ""), 0.05)
    ft_bonus = min(0.25, 0.03 * ft + 0.0005 * max(0, n - 3 * ft - 1))
    latency_pen = {"subsecond": 0.0, "seconds": 0.02, "minutes": 0.05, "hours": 0.08}.get(latency, 0.05)
    attack_pen = 0.04 * len(attacks.intersection({"sybil_attack", "long_range_attack", "eclipse_attack", "nothing_at_stake", "bribery_attack"}))

    score = 0.5 + proto_bonus + ft_bonus - latency_pen - attack_pen
    return float(min(1.0, max(0.0, score)))


def oow_6_risk_pool_var(ax: Dict[str, Any]) -> float:
    """
    OUT-OF-THIS-WORLD #6: Risk Pool Variability (0..1.2).
    Combines pool size, correlation class, and number of categories.
    """
    rp = ax.get("civilization_risk_pooling", {})
    size = int(rp.get("pool_size", 1000))
    corr = rp.get("risk_correlation", "independent")
    kinds = len(rp.get("risk_categories", [])) or 1

    corr_factor = {"independent": 0.6, "weakly_correlated": 0.8, "strongly_correlated": 1.0, "systemic": 1.2}.get(corr, 0.9)
    size_factor = 1.0 / np.log10(max(10, size))
    raw = corr_factor * size_factor * np.sqrt(kinds)
    return float(min(1.2, max(0.0, raw)))


def oow_7_utility_frontier_cloud(ax: Dict[str, Any], samples: int = 250) -> pd.DataFrame:
    """
    OUT-OF-THIS-WORLD #7: Utility frontier point cloud (toy sampler).
    Samples tri-objective values according to declared frontier shape.
    Returns DataFrame with columns o1, o2, o3 and labels for objectives.
    """
    uf = ax.get("civilization_utility_frontier", {})
    objs = uf.get("objectives", ["total_welfare", "resilience_metric", "coordination_efficiency"])
    shape = uf.get("frontier_shape", "convex")

    rng = np.random.default_rng(hash_seed(ax.get("seed_concept", "") + ax.get("timestamp", "")))
    X = rng.random((samples, 3))

    if shape == "convex":
        X = X ** 0.5  # bias towards edges
    elif shape == "concave":
        X = X ** 2.0  # bias towards interior
    elif shape == "linear":
        X[:, 2] = 1 - X[:, 0]  # crude trade-off line + noise
        X[:, 1] = (X[:, 0] + X[:, 2]) / 2 + rng.normal(0, 0.02, size=samples)
    elif shape == "discontinuous":
        mask = X[:, 0] > 0.5
        X[mask] *= 0.5
        X[~mask] = 0.5 + 0.5 * X[~mask]

    df = pd.DataFrame(X, columns=["o1", "o2", "o3"])
    df["labels"] = "|".join(objs[:3])
    return df


def oow_8_institution_evolution_markov(ax: Dict[str, Any]) -> Dict[str, float]:
    """
    OUT-OF-THIS-WORLD #8: One-step Markov prediction of institution evolution stage.
    """
    stage = get(ax, "agi_institution.evolution_stage", "formation")
    stages = ["formation", "consolidation", "maturity", "transformation"]
    i = stages.index(stage) if stage in stages else 0
    probs = np.zeros(len(stages))
    probs[i] = 0.5
    probs[(i + 1) % len(stages)] += 0.35
    probs[(i - 1) % len(stages)] += 0.15
    probs = probs / probs.sum()
    return {stages[k]: float(v) for k, v in enumerate(probs)}


def oow_9_paradox_stability_score(ax: Dict[str, Any]) -> float:
    """
    OUT-OF-THIS-WORLD #9: Paradox Stability Score [0, 1].
    Causal Loop and Temporal gain stability; Linguistic penalized for contradictions.
    """
    ptype = ax.get("paradox_type", "").lower()
    mech = " ".join(ax.get("mechanisms", [])).lower()
    base = {"causal loop": 0.7, "temporal": 0.65, "cosmic": 0.6, "entropic": 0.55, "metaphysical": 0.5, "linguistic": 0.35}.get(ptype, 0.5)
    loop_bonus = 0.05 if "loop" in mech else 0.0
    entropy_pen = 0.04 if "entropy" in mech else 0.0
    return float(min(1.0, max(0.0, base + loop_bonus - entropy_pen)))


def oow_10_mechanism_entropy(all_axioms: List[Dict[str, Any]]) -> float:
    """
    OUT-OF-THIS-WORLD #10: Shannon entropy over mechanism distribution across all axioms.
    """
    from collections import Counter
    counts = Counter()
    for ax in all_axioms:
        for m in ax.get("mechanisms", []):
            counts[m] += 1
    if not counts:
        return 0.0
    p = np.array(list(counts.values()), dtype=float)
    p = p / p.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(H)


def oow_11_phase_transition_probability(ax: Dict[str, Any]) -> Dict[str, float]:
    """
    OUT-OF-THIS-WORLD #11: Logistic probabilities for next civilization phase given indicators.
    """
    phase = ax.get("civilization_phase", "informational")
    indicators = set(ax.get("phase_indicators", []))
    all_phases = ["hunter_gatherer", "agricultural", "industrial", "informational", "post_scarcity"]
    base_idx = all_phases.index(phase) if phase in all_phases else 2
    logits = np.array([-3., -1., 0., 1., 3.])
    # Nudges from indicators
    if "network effects" in indicators or "attention economics" in indicators:
        logits[all_phases.index("informational")] += 1.0
        logits[all_phases.index("post_scarcity")] += 0.5
    if "mass production" in indicators or "specialization" in indicators:
        logits[all_phases.index("industrial")] += 1.0
    if "surplus accumulation" in indicators or "long-term planning" in indicators:
        logits[all_phases.index("agricultural")] += 0.6

    # Prefer near neighbors
    distance = np.array([abs(i - base_idx) for i in range(len(all_phases))], dtype=float)
    logits -= 0.7 * distance
    # Softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return {p: float(probs[i]) for i, p in enumerate(all_phases)}


def oow_12_coevolutionary_pressure(ax: Dict[str, Any]) -> float:
    """
    OUT-OF-THIS-WORLD #12: Coevolutionary Pressure Meter [0, 1].
    Uses intensity and innovation_rate from the arms race.
    """
    ar = ax.get("coevolutionary_arms_race", {})
    intensity = float(ar.get("intensity", 0.5))
    innov = float(ar.get("innovation_rate", 0.1))
    return float(min(1.0, max(0.0, 0.8 * intensity + 0.6 * innov)))


# --------------------------- Plot helpers (Matplotlib) ---------------------

def plot_replicator(df: pd.DataFrame, title: str = "Replicator Dynamics"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in [c for c in df.columns if c != "t"]:
        ax.plot(df["t"], df[col], label=col)
    ax.set_xlabel("time")
    ax.set_ylabel("proportion")
    ax.set_title(title)
    ax.legend(loc="best")
    st.pyplot(fig)


def plot_scaling_curve(df: pd.DataFrame, title: str = "Coordination Scaling Projection"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(df["scale"], df["efficiency"])
    ax.set_xlabel("agents (log scale)")
    ax.set_ylabel("coordination efficiency")
    ax.set_title(title)
    st.pyplot(fig)


def plot_pareto_cloud(df: pd.DataFrame, title: str = "Utility Frontier Cloud"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["o1"], df["o2"], s=10, alpha=0.5)
    ax.set_xlabel("o1")
    ax.set_ylabel("o2")
    ax.set_title(title + " (o1 vs o2)")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(df["o1"], df["o3"], s=10, alpha=0.5)
    ax2.set_xlabel("o1")
    ax2.set_ylabel("o3")
    ax2.set_title(title + " (o1 vs o3)")
    st.pyplot(fig2)


# --------------------------- Streamlit UI ---------------------------

def sidebar_controls():
    st.sidebar.header("Generation Controls")
    generator_path = st.sidebar.text_input("Path to generator", GENERATOR_FILENAME_DEFAULT)
    seed = st.sidebar.text_input("Seed", "Trust emerges when verification costs approach zero and betrayal becomes evolutionarily unstable")
    tone = st.sidebar.selectbox("Tone", ["civilization", "tao", "poetic", "plain", "academic", "oracular"], index=0)
    count = st.sidebar.slider("Count", 1, 24, 8, 1)
    rng = st.sidebar.number_input("RNG (optional)", min_value=0, value=137, step=1)
    max_mech = st.sidebar.slider("Max mechanisms", 1, 8, 5, 1)
    tao = st.sidebar.checkbox("Apply Tao mode", value=False)
    civilization = st.sidebar.checkbox("Apply Civilization mode", value=True)
    math_theme = st.sidebar.selectbox("Math theme", ["auto", "GR", "path-integral"])
    selfref_mode = st.sidebar.selectbox("Self-ref mode", ["meta", "binary", "field"], index=0)
    save_to_history = st.sidebar.checkbox("Save to Civilization_Session.json", value=True)

    st.sidebar.header("Load/Export")
    load_session = st.sidebar.file_uploader("Load Civilization_Session.json", type=["json"])
    export_csv = st.sidebar.checkbox("Allow CSV export", value=True)
    return {
        "generator_path": generator_path,
        "seed": seed,
        "tone": tone,
        "count": count,
        "rng": int(rng),
        "max_mech": max_mech,
        "tao": tao,
        "civilization": civilization,
        "math_theme": math_theme,
        "selfref_mode": selfref_mode,
        "save_to_history": save_to_history,
        "load_session": load_session,
        "export_csv": export_csv,
    }


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Visualize and simulate axioms emitted by the Civilization Horizon generator.")

    controls = sidebar_controls()

    session_axioms: List[Dict[str, Any]] = []
    raw_io_text = ""

    # Load session if provided
    if controls["load_session"] is not None:
        try:
            session_axioms = load_saved_session(Path(controls["load_session"].name))
        except Exception:
            # Streamlit uploads are file-like; read their bytes
            try:
                data = json.loads(controls["load_session"].read().decode("utf-8"))
                session_axioms = data["results"] if isinstance(data, dict) and "results" in data else []
            except Exception as e:
                st.error(f"Failed to load session: {e}")

    colA, colB = st.columns([2, 1])
    with colA:
        if st.button("Generate Axioms", use_container_width=True):
            gen_path = Path(controls["generator_path"]).expanduser().resolve()
            if not gen_path.exists():
                st.error(f"Generator script not found at {gen_path}")
            else:
                axioms, io = run_generator_cli(
                    gen_path,
                    seed=controls["seed"],
                    tone=controls["tone"],
                    count=controls["count"],
                    rng=controls["rng"],
                    max_mech=controls["max_mech"],
                    tao=controls["tao"],
                    civilization=controls["civilization"],
                    math_theme=controls["math_theme"],
                    selfref_mode=controls["selfref_mode"],
                    save_to_history=controls["save_to_history"],
                )
                session_axioms = axioms
                raw_io_text = io

    with colB:
        st.subheader("Quick Stats")
        if session_axioms:
            st.metric("Axioms Loaded", len(session_axioms))
            H_mech = oow_10_mechanism_entropy(session_axioms)
            st.metric("Mechanism Entropy", f"{H_mech:.2f}")
        else:
            st.write("No axioms yet. Generate or load a session.")

    # Export CSV
    if session_axioms and controls["export_csv"]:
        if st.button("Export CSV", use_container_width=True):
            csv_path = Path("axioms_export.csv")
            export_axioms_csv(session_axioms, csv_path)
            with csv_path.open("rb") as f:
                st.download_button(
                    label="Download axioms_export.csv",
                    data=f.read(),
                    file_name="axioms_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Raw IO debug
    with st.expander("Show Raw Generator I/O"):
        st.text_area("stdout/stderr", value=raw_io_text, height=200)

    # Axiom Browser and Simulations
    if session_axioms:
        st.subheader("Axiom Browser")
        options = [f"{i+1:02d} | {ax.get('core_statement','<no core>')}" for i, ax in enumerate(session_axioms)]
        sel = st.selectbox("Select an axiom", options, index=0)
        idx = options.index(sel) if sel in options else 0
        axiom = session_axioms[idx]

        # Overview
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Core Statement**")
            st.info(axiom.get("core_statement", ""))
            st.write("**Paradox Type**:", axiom.get("paradox_type", ""))
            st.write("**Consequences**:", axiom.get("consequences", ""))
            st.write("**Tone/Scaffold**:", axiom.get("humanized_injection_scaffold", ""))
        with c2:
            st.write("**Mechanisms**")
            st.write(", ".join(axiom.get("mechanisms", [])) or "—")
            st.write("**Axiom Text**")
            st.code(axiom.get("axiom_text", ""), language="markdown")
        with c3:
            st.write("**Metrics**")
            st.json(axiom.get("metrics", {}))
            st.write("**Timestamp**:", axiom.get("timestamp", ""))
            st.write("**Seed**:", axiom.get("seed_concept", ""))

        st.divider()
        st.subheader("Simulations & Analytics")

        # OOW #1 Replicator
        st.markdown("**1) Replicator Dynamics (strategy mix over time)**")
        df_rep = oow_1_replicator_dynamics(axiom, steps=140)
        plot_replicator(df_rep)

        # OOW #2 Attention Gauge
        st.markdown("**2) Attention Gauge Flux Score**")
        att = oow_2_attention_gauge(axiom)
        st.metric("Attention Flux", f"{att:.3f}")

        # OOW #3 Trust Emergence
        st.markdown("**3) Trust Emergence Index**")
        trust = oow_3_trust_emergence_index(axiom)
        st.metric("Trust Index", f"{trust:.3f}")

        # OOW #4 Coordination scaling
        st.markdown("**4) Coordination Scaling Projection**")
        df_scale = oow_4_coordination_scaling_curve(axiom)
        plot_scaling_curve(df_scale)

        # OOW #5 Consensus Robustness
        st.markdown("**5) Consensus Robustness Score**")
        cr = oow_5_consensus_robustness(axiom)
        st.metric("Consensus Robustness", f"{cr:.3f}")

        # OOW #6 Risk Pool Variability
        st.markdown("**6) Risk Pool Variability**")
        var = oow_6_risk_pool_var(axiom)
        st.metric("Risk Pool VAR", f"{var:.3f}")

        # OOW #7 Utility Frontier Cloud
        st.markdown("**7) Utility Frontier Cloud (toy sampler)**")
        df_cloud = oow_7_utility_frontier_cloud(axiom, samples=250)
        plot_pareto_cloud(df_cloud, title="Utility Frontier Cloud")

        # OOW #8 Institution Evolution Markov
        st.markdown("**8) Next Institution Stage (one-step Markov)**")
        st.json(oow_8_institution_evolution_markov(axiom))

        # OOW #9 Paradox Stability
        st.markdown("**9) Paradox Stability Score**")
        pss = oow_9_paradox_stability_score(axiom)
        st.metric("Paradox Stability", f"{pss:.3f}")

        # OOW #10 Mechanism Entropy across session
        st.markdown("**10) Session-wide Mechanism Entropy**")
        st.metric("Mechanism Shannon Entropy", f"{oow_10_mechanism_entropy(session_axioms):.3f}")

        # OOW #11 Phase Transition Probabilities
        st.markdown("**11) Civilization Phase Transition Probabilities**")
        st.json(oow_11_phase_transition_probability(axiom))

        # OOW #12 Coevolutionary Pressure
        st.markdown("**12) Coevolutionary Pressure Meter**")
        st.metric("Pressure", f"{oow_12_coevolutionary_pressure(axiom):.3f}")

        st.divider()

        # Save mini-session (selected axiom only) for downstream tooling
        if st.button("Save Selected Axiom to JSON", use_container_width=True):
            out = Path(f"axiom_selected_{idx+1:02d}.json")
            out.write_text(json.dumps(axiom, ensure_ascii=False, indent=2), encoding="utf-8")
            with out.open("rb") as f:
                st.download_button("Download selected axiom JSON", f.read(), file_name=out.name, mime="application/json", use_container_width=True)


if __name__ == "__main__":
    main()
