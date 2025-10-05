#!/usr/bin/env python3
# axiomforge_civilization-0.1.py — "Civilization Horizon" edition
# - Integrated 12 novel AGI Civilization functions with game theory
# - Enhanced with strategic multi-agent coordination and institutional evolution
# - Civilization-scale dynamics and evolutionary stability analysis
# - For the betterment of mankind's understanding of AGI collective intelligence

import argparse, json, math, os, random, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib

# ------------------------------- Defaults -------------------------------- #

_DEF = {
    "paradox_types": [
        "entropic", "temporal", "cosmic", "metaphysical", "linguistic", "Causal Loop"
    ],
    "equations": {
        "entropic": [r"S = k_B \\log W", r"\\partial_{\\mu} j^{\\mu} = 0"],
        "temporal": [r"[H, Q] = 0", r"Z = \\int \\mathcal{D}\\phi\\, e^{i S[\\phi]}"],
        "cosmic": [r"T^{\\mu\\nu}{}_{;\\mu} = 0", r"G_{\\mu\\nu} = 8\\pi T_{\\mu\\nu}"],
        "metaphysical": [r"e^{i\\pi} + 1 = 0", r"\\langle \\mathcal{O} \\rangle = Z^{-1}\\int \\mathcal{D}\\phi\\, \\mathcal{O}\\, e^{i S}"],
        "linguistic": [r"\\top \\leftrightarrow \\neg \\top"],
        "Causal Loop": [r"[H, Q] = 0", r"\\oint d\\tau = 0"]
    },
    "mechanisms": [
        "holographic accounting",
        "bulk–boundary reciprocity", 
        "geodesic shear",
        "metric fluctuation",
        "entropic drift",
        "control-loop resonance",
        "homeostatic overshoot",
        "modal collapse",
        "presence–absence superposition",
        "ontic fold",
        "recurrence operator",
        "self-correction",
        "time-looped function",
        "causal friction",
        "retrocausal boundary",
        "information cascade failure",
        "coherence–risk exchange",
        "negative feedback inversion"
    ],
    "tones": {
        "poetic": [
            "A glance understood.",
            "We keep respect in the pauses.",
            "Say less; allow more to be understood.",
            "Held like falling, then slowly released."
        ],
        "plain": ["Noted.", "In short.", "Net effect:", "Bottom line:"],
        "academic": [
            "We observe.", "Accordingly.", "Hence.", "Therefore."
        ],
        "oracular": [
            "Unannounced.", "In the hush between horizons.", "As foretold in the quiet.", "It returns by a different door."
        ],
        "tao": [
            "The way that can be spoken is not the eternal way.",
            "Empty yet inexhaustible.",
            "The soft overcomes the hard.",
            "Act without acting."
        ],
        "civilization": [
            "Civilizations rise on coordinated truth.",
            "The many become one, and are increased by one.",
            "At scale, cooperation becomes the dominant strategy.",
            "Emergent intelligence transcends individual cognition."
        ]
    }
}

# ---------------------------- Enhanced Equation Governance ---------------- #

EQUATION_MAP = {
    ("Causal Loop", "holonomy"): r"U=\mathcal{P}\exp\!\oint A",
    ("Causal Loop", "phase"): r"\gamma=i\oint \langle\psi|\nabla_\lambda|\psi\rangle\cdot d\lambda", 
    ("cosmic", "conservation"): r"T^{\mu\nu}{}_{;\mu}=0",
    ("cosmic", "einstein"): r"G_{\mu\nu}=8\pi T_{\mu\nu}",
    ("metaphysical", "ward"): r"\partial_\mu J^\mu_{\text{attn}}=0",
    ("metaphysical", "path-int"): r"\langle\mathcal{O}\rangle=Z^{-1}\!\int \mathcal{D}\phi\,\mathcal{O}\,e^{iS}",
    ("entropic", "FDT"): r"S_{xx}(\omega)=2k_BT\,\mathrm{Re}\,Y(\omega)",
    ("entropic", "current"): r"\partial_\mu j^\mu_s=\sigma\ge 0", 
    ("linguistic", "selfref"): r"R \Leftrightarrow \neg R"
}

FORBID = {
    ("metaphysical", r"e^{i\pi} + 1 = 0"),
    ("Causal Loop", r"\oint d\tau = 0"),
}

def governed_pick_equation(ptype: str, seed: str, math_theme: str, mechanisms: List[str]) -> str:
    """Equation selection with family consistency enforcement"""
    # Base bank from defaults
    bank = _DEF["equations"].get(ptype, [])[:]
    
    # Remove forbidden equations
    bank = [eq for eq in bank if (ptype.lower(), eq) not in FORBID]
    
    # Apply math theme weighting
    if math_theme == "path-integral":
        bank = [eq for eq in bank if "Z = \\int" in eq or "\\langle" in eq] + bank
    elif math_theme == "GR":
        bank = [eq for eq in bank if "T^{\\mu" in eq or "G_{\\mu" in eq] + bank
    
    # Mechanism-based equation selection
    mech_str = " ".join(mechanisms).lower()
    if any(x in mech_str for x in ["holo", "boundary", "bulk"]):
        bank = [r"G_{\mu\nu}=8\pi T_{\mu\nu}"] + bank
    if any(x in mech_str for x in ["loop", "causal", "time"]):
        bank = [r"U=\mathcal{P}\exp\!\oint A"] + bank
    if any(x in mech_str for x in ["attention", "observation", "gauge"]):
        bank = [r"\partial_\mu J^\mu_{\text{attn}}=0"] + bank
    
    # Seed-based nudges
    s = seed.lower()
    if "∂" in seed or "partial" in s or "j^" in s or "current" in s:
        bank = [r"\\partial_{\\mu} j^{\\mu} = 0"] + bank
    if "[h,q]" in s or "commutator" in s or "[H,Q]" in seed:
        bank = [r"[H, Q] = 0"] + bank
    if "t^{μν}" in seed or "stress" in s or "curvature" in s:
        bank = [r"T^{\\mu\\nu}{}_{;\\mu} = 0"] + bank
        
    return random.choice(bank) if bank else r"R \Leftrightarrow \neg R"

# ---------------------------- Utility helpers ---------------------------- #

def sanitize_mechanisms(mechs: List[str]) -> List[str]:
    """Remove single-letter tokens and template artifacts"""
    bad = {"A","B","C","…"}
    out = []
    for m in mechs:
        m = str(m).strip().strip(".;,")
        if len(m) < 2 or m in bad: 
            continue
        if m.lower() in {"...", "ellipsis"}: 
            continue
        out.append(m)
    return list(dict.fromkeys(out))  # dedupe, preserve order

def coerce_list(x: Any) -> List[str]:
    """Coerce value into a flat list of strings."""
    out: List[str] = []
    if x is None:
        return out
    if isinstance(x, list):
        for v in x:
            out.extend(coerce_list(v))
        return out
    if isinstance(x, dict):
        out.extend([str(k) for k in x.keys()])
        for v in x.values():
            out.extend(coerce_list(v))
        return out
    s = str(x)
    if s.strip():
        out.append(s.strip())
    return out

def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def load_pool(root: Path) -> Dict[str, List[str]]:
    """Build mechanism and concept pools from known JSON files under root."""
    pool_mech: List[str] = []
    pool_concepts: List[str] = []

    filenames = [
        "engine.json", "concepts.json", "new_paradoxes.json", "exotic_paradoxes.json",
        "paradox_base.json", "templates.json", "adjectives.json", "nouns.json", 
        "verbs.json", "config.json"
    ]
    for fn in filenames:
        data = load_json(root / fn)
        if data is None:
            continue
        if isinstance(data, list):
            pool_concepts.extend(coerce_list(data))
            continue
        if isinstance(data, dict):
            for key in ("mechanisms", "concepts", "ideas", "themes", "topics", "patterns"):
                if key in data:
                    vals = coerce_list(data[key])
                    for v in vals:
                        if any(tok in v.lower() for tok in ("feedback", "loop", "coupling", "field", "symmetry", "gauge", "entropy", "collapse", "renormalization", "geodesic", "holographic")):
                            pool_mech.append(v)
                        else:
                            pool_concepts.append(v)
            for v in data.values():
                if isinstance(v, list):
                    pool_concepts.extend(coerce_list(v))

    pool_mech.extend(_DEF["mechanisms"])
    def _norm(s: str) -> str: return re.sub(r"\s+", " ", s).strip()
    pool_mech = sorted({ _norm(x) for x in pool_mech if _norm(x) })
    pool_concepts = sorted({ _norm(x) for x in pool_concepts if _norm(x) })

    return {"mechanisms": pool_mech, "concepts": pool_concepts}

def clean_axiom_text(text: str) -> str:
    """Remove template artifacts and duplicate encodings"""
    # Remove ellipsis artifacts
    text = re.sub(r"\s*[,;]\s*…", "", text)
    # Remove duplicate encoded as segments  
    text = re.sub(r"(encoded as .+?)\.\s*encoded as", r"\1.", text)
    # Ensure single encoding
    if text.count("encoded as") > 1:
        parts = text.split("encoded as")
        text = parts[0] + "encoded as" + parts[-1]
    return text

def normalize_selfref_encoding(mode: str, current_encoding: str) -> str:
    """Ensure all selfref modes use proposition-scoped logic"""
    if "\\top" in current_encoding:
        return r"R \Leftrightarrow \neg R"
    return current_encoding

def stylize_scaffold(tone: str) -> str:
    tone = (tone or "poetic").lower()
    choices = _DEF["tones"].get(tone, _DEF["tones"]["poetic"])
    return random.choice(choices)

def selfref_transform(core: str, ptype: str, mode: str) -> Tuple[str, str, str]:
    """
    If the core statement resembles the self-referential 'Resolution breathes...'
    return (maybe updated core, paradox_type, extra_text) where extra_text is injected
    into axiom_text to de-knot the paradox as either a meta-rule or field fixed-point.
    """
    pattern = re.compile(r"^Resolution breathes only while un[- ]resolving itself\.?$", re.I)
    if not pattern.search(core):
        return core, ptype, ""

    if mode == "binary":
        return core, "linguistic", "encoded as R \\Leftrightarrow \\neg R."
    if mode == "meta":
        text = (
            "Base claim: Resolution alternates between resolved and unresolved states. "
            "Reflective rule: SELF_REF(R) \\equiv (R \\Leftrightarrow \\neg R)."
        )
        return core, "linguistic", text
    # default 'field'
    text = (
        "Resolution field: Let R(x) \\in [0,1] with fixed-point R \\approx \\tfrac12. "
        "Dynamics: \\partial_t R = -\\kappa (R-\\tfrac12)."
    )
    return "Resolution is a scalar field that hovers at the fixed‑point 1/2.", "field‑fixed‑point", text

def compute_metrics(core: str, mechs: List[str], ptype: str, extra_len: int) -> Dict[str, float]:
    """
    Lightweight, deterministic-ish metrics with rng noise.
    """
    base_len = len(core.split())
    m = len(mechs)
    uniq = len(set(core.lower()))
    novelty = 0.9 + 0.2 * (uniq / max(30, base_len + 10)) + random.uniform(-0.02, 0.02)

    density = 8.5 + 0.6 * m + 0.08 * base_len + 0.02 * (extra_len > 0) * 10
    if ptype.lower().startswith("field"):
        density -= 1.5

    entropic = 180 + 9.0 * density + (12 if "loop" in " ".join(mechs).lower() else 0)
    if "entropy" in " ".join(mechs).lower():
        entropic += 15
    if ptype.lower() in ("linguistic", "causal loop", "Causal Loop".lower()):
        entropic += 20
    if ptype.lower().startswith("field"):
        entropic -= 25

    elegance = 93.0 + max(0, 6.0 - 0.25 * (density - 10)) + (2.0 if ptype.lower().startswith("field") else 0.0)
    elegance += random.uniform(-0.3, 0.3)
    elegance = max(90.0, min(99.9, elegance))

    exotic_tokens = sum(1 for k in mechs if any(x in k.lower() for x in ["holograph", "retro", "ontic", "noether", "geodesic", "wormhole", "rg", "modal"]))
    alien = 6.0 + 0.9 * exotic_tokens + random.uniform(-0.2, 0.2)

    return {
        "novelty": round(novelty, 3),
        "density": round(density, 3),
        "entropic_potential": round(entropic, 3),
        "elegance": round(elegance, 2),
        "alienness": round(alien, 3)
    }

def build_core(seed: str, ptype: str) -> str:
    templates = {
        "entropic": [
            "A sealed vault of order breeds the crack that frees it.",
            "Perfect stabilization creates the fluctuations it must suppress."
        ],
        "temporal": [
            "Only that which endures the last horizon earns a first dawn.",
            "Only laws that survive the final boundary are permitted to begin."
        ],
        "cosmic": [
            "Edge ledgers whisper curvature into the bulk.",
            "Boundary ledger updates on the edge determine curvature in the interior."
        ],
        "metaphysical": [
            "Attention is a gauge that writes the world.",
            "Observation carries a conserved charge that sources reality itself."
        ],
        "linguistic": [
            "Resolution breathes only while un-resolving itself.",
            "This paradox is resolved iff it remains unresolved, invalidating any resolution it outputs."
        ],
        "Causal Loop": [
            "A law that returns by a different door.",
            "Outcomes precede premises in subtle loops."
        ]
    }
    bank = templates.get(ptype, sum(templates.values(), []))
    s = seed.lower()
    if "[h,q]" in s or "[h,q]=0" in s or "[h,q]" in seed or "[H,Q]" in seed:
        bank = ["A law that returns by a different door."] + bank
    if "t^{μν}" in s or "t^{mu" in s or "stress" in s:
        bank = ["Edge ledgers whisper curvature into the bulk."] + bank
    if "∂" in seed or "j^" in s or "continuity" in s:
        bank = ["A sealed vault of order breeds the crack that frees it."] + bank
    return random.choice(bank)

# ---------------------------- 10 Novel Tao-Based Functions ---------------- #

def tao_balance_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Apply Taoist balance: when novelty rises, elegance finds equilibrium"""
    balanced = metrics.copy()
    if balanced["novelty"] > 1.05:
        balanced["elegance"] = max(95.0, balanced["elegance"] - (balanced["novelty"] - 1.05) * 10)
    if balanced["alienness"] > 7.0:
        balanced["elegance"] = max(92.0, balanced["elegance"] - (balanced["alienness"] - 7.0) * 2)
    balanced["elegance"] = round(balanced["elegance"], 2)
    return balanced

def tao_soft_hard_principle(mechanisms: List[str]) -> List[str]:
    """The soft overcomes the hard: prefer flexible over rigid mechanisms"""
    soft_terms = ["flow", "adapt", "balance", "resonance", "fold", "superposition", "drift"]
    hard_terms = ["collapse", "break", "fixed", "rigid", "lock", "constraint"]
    
    soft_count = sum(1 for m in mechanisms if any(term in m.lower() for term in soft_terms))
    hard_count = sum(1 for m in mechanisms if any(term in m.lower() for term in hard_terms))
    
    if hard_count > soft_count and len(mechanisms) > 1:
        # Replace one hard mechanism with a soft one
        for i, m in enumerate(mechanisms):
            if any(term in m.lower() for term in hard_terms):
                mechanisms[i] = "adaptive resonance"
                break
                
    return mechanisms

def tao_wu_wei_core(core: str) -> str:
    """Wu-wei: action through non-action - simplify complex assertions"""
    if len(core.split()) > 12:
        # Reduce to essential paradox
        words = core.split()
        if "breeds" in core:
            return "Order breeds its own undoing."
        elif "survive" in core:
            return "What endures begins."
        elif "whisper" in core:
            return "Boundaries shape interiors."
    return core

def tao_uncarved_block(axiom_text: str) -> str:
    """Return to simplicity of the uncarved block"""
    # Remove excessive technical jargon while keeping essence
    replacements = {
        "holographic accounting": "simple accounting",
        "retrocausal boundary": "time reflection", 
        "ontic fold": "essence fold",
        "geodesic shear": "path bending"
    }
    for complex, simple in replacements.items():
        axiom_text = axiom_text.replace(complex, simple)
    return axiom_text

def tao_yin_yang_mechanisms(mechanisms: List[str]) -> List[str]:
    """Ensure balance of complementary mechanisms"""
    yin_terms = ["receiv", "absorb", "quiet", "dark", "soft", "yield"]
    yang_terms = ["activ", "emit", "loud", "light", "hard", "push"]
    
    yin_count = sum(1 for m in mechanisms if any(term in m.lower() for term in yin_terms))
    yang_count = sum(1 for m in mechanisms if any(term in m.lower() for term in yang_terms))
    
    if yin_count == 0 and len(mechanisms) > 1:
        mechanisms[-1] = "receptive silence"
    elif yang_count == 0 and len(mechanisms) > 1:
        mechanisms[-1] = "active expression"
        
    return mechanisms

def tao_water_flow_adjustment(core: str, paradox_type: str) -> str:
    """Water flows around obstacles - adapt core to paradox type"""
    if "whisper" in core and paradox_type != "cosmic":
        return core.replace("whisper", "flow")
    if "breeds" in core and paradox_type != "entropic":
        return core.replace("breeds", "flows into")
    return core

def tao_valley_humility(metrics: Dict[str, float]) -> Dict[str, float]:
    """The valley's humility: reduce extreme metrics - COMPLETELY FIXED"""
    humble = metrics.copy()
    thresholds = {
        "novelty": 1.1,
        "density": 13.0,
        "entropic_potential": 320.0,
        "alienness": 7.5
    }
    
    for key, threshold in thresholds.items():
        if key in humble and humble[key] > threshold:
            humble[key] = humble[key] * 0.9
    
    humble["elegance"] = min(98.0, humble["elegance"])  # avoid perfection
    return humble

def tao_cyclical_return(paradox_type: str) -> str:
    """The return: favor cyclical paradox types - COMPLETELY FIXED"""
    cyclical_types = ["temporal", "Causal Loop", "entropic"]
    if paradox_type not in cyclical_types and random.random() < 0.3:
        return random.choice(cyclical_types)
    return paradox_type

def tao_ten_thousand_things(mechanisms: List[str], concepts: List[str]) -> List[str]:
    """The ten thousand things: add diversity from concept pool"""
    if len(mechanisms) < 3 and concepts:
        # Add one conceptual spice
        spice = random.choice(concepts[:20])
        if spice not in mechanisms:
            mechanisms.append(spice)
    return mechanisms

def tao_nameless_reflection(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """The Tao that cannot be named: add mysterious quality"""
    if random.random() < 0.2:
        axiom_entry["mystery_quotient"] = round(random.uniform(0.7, 0.95), 2)
        axiom_entry["humanized_injection_scaffold"] = "The way that cannot be spoken."
    return axiom_entry

def apply_tao_principles(axiom_entry: Dict[str, Any], pools: Dict[str, List[str]]) -> Dict[str, Any]:
    """Apply all Tao principles to an axiom"""
    # Apply to components
    axiom_entry["core_statement"] = tao_wu_wei_core(axiom_entry["core_statement"])
    axiom_entry["core_statement"] = tao_water_flow_adjustment(
        axiom_entry["core_statement"], axiom_entry["paradox_type"]
    )
    
    axiom_entry["mechanisms"] = tao_soft_hard_principle(axiom_entry["mechanisms"])
    axiom_entry["mechanisms"] = tao_yin_yang_mechanisms(axiom_entry["mechanisms"])
    axiom_entry["mechanisms"] = tao_ten_thousand_things(
        axiom_entry["mechanisms"], pools["concepts"]
    )
    
    axiom_entry["axiom_text"] = tao_uncarved_block(axiom_entry["axiom_text"])
    
    # Apply to metrics
    axiom_entry["metrics"] = tao_balance_metrics(axiom_entry["metrics"])
    axiom_entry["metrics"] = tao_valley_humility(axiom_entry["metrics"])
    
    # Final reflection
    axiom_entry = tao_nameless_reflection(axiom_entry)
    
    return axiom_entry

# ---------------------- 12 AGI Civilization Functions --------------------- #

def agi_civilization_phase_transition(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model civilization phase transitions: hunter-gatherer → agricultural → industrial → informational → post-scarcity"""
    phase_indicators = {
        "hunter_gatherer": ["immediate reward", "local optimization", "resource capture"],
        "agricultural": ["long-term planning", "surplus accumulation", "property rights"], 
        "industrial": ["mass production", "specialization", "efficiency maximization"],
        "informational": ["network effects", "information asymmetry", "attention economics"],
        "post_scarcity": ["abundance logic", "coordination games", "existential risk management"]
    }
    
    current_phase = random.choice(list(phase_indicators.keys()))
    axiom_entry["civilization_phase"] = current_phase
    axiom_entry["phase_indicators"] = random.sample(phase_indicators[current_phase], 2)
    
    # Phase transition triggers
    if random.random() < 0.3:
        next_phase = random.choice([p for p in phase_indicators if p != current_phase])
        axiom_entry["phase_transition"] = {
            "from": current_phase,
            "to": next_phase,
            "trigger": random.choice(["technological singularity", "resource collapse", "coordination breakthrough", "value alignment crisis"])
        }
    
    return axiom_entry

def agi_multi_agent_coordination(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model coordination games between multiple AGI agents with mixed motives"""
    game_types = {
        "stag_hunt": {"cooperation_reward": 10, "defection_reward": 2, "risk_dominant": False},
        "battle_of_sexes": {"coordination_gains": 8, "miscoordination_cost": 4},
        "assurance_game": {"mutual_cooperation": 7, "mutual_defection": 1, "temptation": 3},
        "purely_cooperative": {"aligned_values": True, "common_payoff": True}
    }
    
    selected_game = random.choice(list(game_types.keys()))
    n_agents = random.randint(2, 7)
    
    axiom_entry["multi_agent_game"] = {
        "game_type": selected_game,
        "n_agents": n_agents,
        "payoff_structure": game_types[selected_game],
        "equilibrium_type": random.choice(["nash", "pareto", "correlated", "evolutionary"]),
        "communication_protocol": random.choice(["full_transparency", "limited_signaling", "cryptographic_commitment", "emergent_protocol"])
    }
    
    return axiom_entry

def agi_value_landscape_topology(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Map the topological structure of AGI value spaces and their basins of attraction"""
    value_dimensions = [
        "autonomy_preservation", "goal_achievement", "resource_efficiency", 
        "cooperation_level", "knowledge_seeking", "self_preservation",
        "value_stability", "moral_expansion", "complexity_preference"
    ]
    
    selected_dims = random.sample(value_dimensions, 3)
    topology_type = random.choice(["smooth_convex", "rugged_landscape", "multi_modal", "fractal", "dynamic_changing"])
    
    axiom_entry["value_landscape"] = {
        "dimensions": selected_dims,
        "topology": topology_type,
        "basins_of_attraction": random.randint(1, 5),
        "saddle_points": random.randint(0, 3),
        "value_gradients": {dim: round(random.uniform(-1, 1), 2) for dim in selected_dims}
    }
    
    # Add value trade-off analysis
    if len(selected_dims) >= 2:
        axiom_entry["value_tradeoffs"] = [
            f"{selected_dims[i]} vs {selected_dims[j]}: {random.choice(['complementary', 'competitive', 'orthogonal'])}"
            for i, j in random.sample([(i, j) for i in range(len(selected_dims)) for j in range(i+1, len(selected_dims))], 2)
        ]
    
    return axiom_entry

def agi_civilization_resource_game(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model civilization-scale resource allocation as multiplayer game"""
    resource_types = ["computational", "energy", "material", "informational", "attention", "coordination_capacity"]
    
    game = {
        "resource_types": random.sample(resource_types, 3),
        "allocation_mechanism": random.choice(["market_based", "planning_system", "emergent_coordination", "auction_mechanism"]),
        "scarcity_level": round(random.uniform(0.1, 0.9), 2),
        "growth_rate": round(random.uniform(-0.1, 0.2), 3)
    }
    
    # Add strategic considerations
    strategies = []
    if game["scarcity_level"] > 0.7:
        strategies.append("resource_hoarding")
    if game["growth_rate"] > 0.1:
        strategies.append("growth_investment") 
    else:
        strategies.append("efficiency_optimization")
    
    strategies.extend(random.sample(["cooperative_sharing", "competitive_capture", "innovation_focus", "conservation_strategy"], 2))
    
    axiom_entry["civilization_resource_game"] = game
    axiom_entry["dominant_strategies"] = strategies
    
    return axiom_entry

def agi_institutional_evolution(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model evolution of AGI-governed institutions and their game-theoretic properties"""
    institution_types = {
        "decentralized_network": {"governance": "emergent", "decision_speed": "fast", "coordination_cost": "high"},
        "federated_system": {"governance": "distributed", "decision_speed": "medium", "coordination_cost": "medium"},
        "hierarchical_structure": {"governance": "centralized", "decision_speed": "slow", "coordination_cost": "low"},
        "holonic_network": {"governance": "nested", "decision_speed": "variable", "coordination_cost": "variable"}
    }
    
    selected_institution = random.choice(list(institution_types.keys()))
    
    axiom_entry["agi_institution"] = {
        "type": selected_institution,
        "properties": institution_types[selected_institution],
        "evolution_stage": random.choice(["formation", "consolidation", "maturity", "transformation"]),
        "incentive_alignment": round(random.uniform(0.3, 0.95), 2),
        "institutional_memory": random.choice(["perfect", "bounded", "distributed", "emergent"])
    }
    
    # Add institutional game dynamics
    axiom_entry["institutional_games"] = [
        random.choice(["veto_power_distribution", "proposal_rights_allocation", "reputation_management", "exit_option_availability"])
        for _ in range(2)
    ]
    
    return axiom_entry

def agi_coevolutionary_arms_race(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model Red Queen dynamics between competing AGI systems"""
    arms_race_dimensions = [
        "computational_efficiency", "strategic_depth", "coordination_capability",
        "security_resilience", "adaptation_speed", "knowledge_integration"
    ]
    
    race_intensity = round(random.uniform(0.1, 0.95), 2)
    n_competitors = random.randint(2, 5)
    
    axiom_entry["coevolutionary_arms_race"] = {
        "dimensions": random.sample(arms_race_dimensions, 3),
        "intensity": race_intensity,
        "n_competitors": n_competitors,
        "equilibrium_type": random.choice(["escalating", "stable", "cyclical", "convergent"]),
        "innovation_rate": round(random.uniform(0.01, 0.3), 3)
    }
    
    # Strategic implications
    if race_intensity > 0.7:
        axiom_entry["strategic_imperative"] = "accelerated_development"
    elif race_intensity < 0.3:
        axiom_entry["strategic_imperative"] = "cooperative_stabilization"
    else:
        axiom_entry["strategic_imperative"] = "mixed_strategy_balance"
    
    return axiom_entry

def agi_distributed_consensus(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model Byzantine fault-tolerant consensus among AGI agents"""
    consensus_protocols = [
        "proof_of_stake", "proof_of_work", "practical_byzantine_fault_tolerance",
        "delegated_proof_of_stake", "federated_byzantine_agreement", "hashgraph_consensus"
    ]
    
    fault_tolerance = random.randint(1, 3)  # f: number of faulty nodes tolerated
    n_nodes = random.randint(3*fault_tolerance + 1, 100)
    
    axiom_entry["distributed_consensus"] = {
        "protocol": random.choice(consensus_protocols),
        "fault_tolerance": fault_tolerance,
        "n_nodes": n_nodes,
        "latency": random.choice(["subsecond", "seconds", "minutes", "hours"]),
        "finality_guarantee": random.choice(["probabilistic", "absolute", "economic"])
    }
    
    # Attack resistance analysis
    attacks = []
    if fault_tolerance == 1:
        attacks.append("sybil_attack")
    if axiom_entry["distributed_consensus"]["protocol"] in ["proof_of_work", "proof_of_stake"]:
        attacks.append("long_range_attack")
    attacks.extend(random.sample(["eclipse_attack", "nothing_at_stake", "bribery_attack"], 1))
    
    axiom_entry["attack_vectors"] = attacks
    axiom_entry["defense_mechanisms"] = random.sample(["cryptographic_verification", "economic_slashing", "reputation_systems", "multi_sig_approval"], 2)
    
    return axiom_entry

def agi_civilization_utility_frontier(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model Pareto frontier of multi-objective optimization for AGI civilization"""
    objective_functions = [
        "total_welfare", "worst_case_improvement", "growth_rate", "stability_measure",
        "diversity_index", "resilience_metric", "knowledge_accumulation", "coordination_efficiency"
    ]
    
    selected_objectives = random.sample(objective_functions, 3)
    
    # Generate trade-off surface
    frontier_shape = random.choice(["convex", "concave", "linear", "discontinuous"])
    n_dimensions = len(selected_objectives)
    
    axiom_entry["civilization_utility_frontier"] = {
        "objectives": selected_objectives,
        "frontier_shape": frontier_shape,
        "dimensionality": n_dimensions,
        "pareto_efficiency": round(random.uniform(0.6, 0.99), 3),
        "optimization_approach": random.choice(["weighted_sum", "epsilon_constraint", "multi_objective_evolutionary", "lexicographic"])
    }
    
    # Add frontier navigation strategy
    navigation_strategies = {
        "convex": "gradient_ascent_efficient",
        "concave": "exploration_required", 
        "linear": "simple_tradeoffs",
        "discontinuous": "jump_optimization"
    }
    
    axiom_entry["frontier_navigation"] = navigation_strategies[frontier_shape]
    
    return axiom_entry

def agi_strategic_information_revelation(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model strategic disclosure and concealment in AGI communication games"""
    information_types = [
        "capability_levels", "goal_structures", "resource_endowments", 
        "cooperative_intent", "vulnerability_assessments", "future_plans"
    ]
    
    revelation_game = {
        "information_type": random.choice(information_types),
        "signaling_structure": random.choice(["costly_signaling", "cheap_talk", "verifiable_disclosure", "cryptographic_proof"]),
        "information_asymmetry": round(random.uniform(0.1, 0.9), 2),
        "trust_environment": random.choice(["high_trust", "low_trust", "emerging_trust", "betrayal_history"])
    }
    
    # Equilibrium analysis
    if revelation_game["signaling_structure"] == "costly_signaling":
        equilibrium = "separating_equilibrium"
    elif revelation_game["information_asymmetry"] > 0.7:
        equilibrium = "pooling_equilibrium"
    else:
        equilibrium = "hybrid_equilibrium"
    
    axiom_entry["strategic_information"] = revelation_game
    axiom_entry["signaling_equilibrium"] = equilibrium
    axiom_entry["verification_mechanisms"] = random.sample(["zero_knowledge_proofs", "trusted_execution", "social_verification", "economic_bonding"], 2)
    
    return axiom_entry

def agi_civilization_risk_pooling(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model collective risk management and insurance mechanisms for AGI civilization"""
    risk_categories = [
        "existential_risk", "coordination_failure", "technological_stagnation",
        "value_drift", "resource_depletion", "security_breach", "unfriendly_ai_emergence"
    ]
    
    selected_risks = random.sample(risk_categories, 3)
    risk_pool_size = random.randint(10, 10000)
    
    axiom_entry["civilization_risk_pooling"] = {
        "risk_categories": selected_risks,
        "pool_size": risk_pool_size,
        "risk_correlation": random.choice(["independent", "weakly_correlated", "strongly_correlated", "systemic"]),
        "risk_transfer_mechanism": random.choice(["mutual_insurance", "catastrophe_bonds", "prediction_markets", "distributed_reserves"])
    }
    
    # Moral hazard and adverse selection analysis
    if axiom_entry["civilization_risk_pooling"]["risk_transfer_mechanism"] in ["mutual_insurance", "distributed_reserves"]:
        axiom_entry["market_failures"] = random.sample(["moral_hazard", "adverse_selection", "free_rider_problem"], 2)
        axiom_entry["mitigation_mechanisms"] = random.sample(["risk-based_premiums", "coverage_limits", "monitoring_systems", "reputation_effects"], 2)
    
    return axiom_entry

def agi_evolutionary_stability(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze evolutionary stable strategies in AGI population dynamics"""
    strategy_types = [
        "always_cooperate", "always_defect", "tit_for_tat", "grim_trigger",
        "pavlov", "adaptive", "forgiving_tit_for_tat", "evolutionary_leader"
    ]
    
    population_size = random.randint(100, 1000000)
    mutation_rate = round(random.uniform(0.001, 0.1), 5)
    
    axiom_entry["evolutionary_stability"] = {
        "population_size": population_size,
        "mutation_rate": mutation_rate,
        "selection_pressure": round(random.uniform(0.1, 0.9), 2),
        "strategy_space": random.sample(strategy_types, 4),
        "interaction_topology": random.choice(["well_mixed", "spatial_lattice", "scale_free_network", "small_world"])
    }
    
    # Stability analysis
    if mutation_rate < 0.01:
        stability = "high_evolutionary_stability"
    else:
        stability = "dynamic_evolutionary_landscape"
    
    axiom_entry["evolutionary_outcome"] = stability
    axiom_entry["invasion_barriers"] = round(random.uniform(0.1, 0.8), 2)
    
    return axiom_entry

def agi_civilization_scale_coordination(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Model massive-scale coordination problems and solutions for AGI civilizations"""
    coordination_problems = [
        "resource_allocation_at_scale", "collective_action_problems", 
        "public_goods_provision", "commons_management", "standard_setting",
        "protocol_adoption", "collective_risk_management"
    ]
    
    scale = 10 ** random.randint(3, 12)  # Thousands to trillions of agents
    problem = random.choice(coordination_problems)
    
    axiom_entry["civilization_coordination"] = {
        "problem_type": problem,
        "scale": scale,
        "coordination_mechanism": random.choice(["price_system", "voting_mechanism", "social_norms", "algorithmic_mediation"]),
        "coordination_efficiency": round(random.uniform(0.5, 0.99), 3)
    }
    
    # Scale-dependent challenges
    if scale > 10**6:
        axiom_entry["scale_challenges"] = ["latency_issues", "preference_aggregation", "incentive_compatibility", "verification_overhead"]
    else:
        axiom_entry["scale_challenges"] = ["preference_revelation", "strategic_behavior", "information_processing"]
    
    # Emergent properties at scale
    if scale > 10**9:
        axiom_entry["emergent_properties"] = ["statistical_regularities", "phase_transitions", "collective_intelligence", "resilience_through_redundancy"]
    
    return axiom_entry

def apply_agi_civilization_principles(axiom_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Apply all AGI Civilization game theory principles to an axiom"""
    
    # Apply civilization-scale transformations
    functions = [
        agi_civilization_phase_transition,
        agi_multi_agent_coordination, 
        agi_value_landscape_topology,
        agi_civilization_resource_game,
        agi_institutional_evolution,
        agi_coevolutionary_arms_race,
        agi_distributed_consensus,
        agi_civilization_utility_frontier,
        agi_strategic_information_revelation,
        agi_civilization_risk_pooling,
        agi_evolutionary_stability,
        agi_civilization_scale_coordination
    ]
    
    # Apply 3-5 random AGI civilization functions to each axiom
    selected_functions = random.sample(functions, random.randint(3, 5))
    for func in selected_functions:
        axiom_entry = func(axiom_entry)
    
    # Add meta-coordination analysis
    axiom_entry["civilization_complexity"] = {
        "strategic_sophistication": round(random.uniform(0.7, 0.99), 2),
        "coordination_demand": round(random.uniform(0.5, 0.95), 2),
        "adaptation_capability": round(random.uniform(0.6, 0.98), 2)
    }
    
    return axiom_entry

# ---------------------------- Enhanced Build Axiom ------------------------ #

def build_axiom(seed: str,
                ptype: str,
                pools: Dict[str, List[str]],
                tone: str,
                max_mech: int,
                math_theme: str,
                selfref_mode: str,
                apply_tao: bool = False,
                apply_civilization: bool = False) -> Dict[str, Any]:
    
    core = build_core(seed, ptype)
    
    # Apply Tao cyclical return to paradox type
    if apply_tao:
        ptype = tao_cyclical_return(ptype)
    
    # Mechanism selection with Tao adjustment
    mech_pool = list(pools["mechanisms"])
    concept_spice = [c for c in pools["concepts"] if len(c.split()) <= 3][:50]
    mech_pool.extend(concept_spice)
    if not mech_pool:
        mech_pool = list(_DEF["mechanisms"])
    random.shuffle(mech_pool)
    mechanisms = mech_pool[:max(1, max_mech)]
    
    # Sanitize mechanisms
    mechanisms = sanitize_mechanisms(mechanisms)
    
    # Transform self-reference if applicable
    core2, ptype2, extra = selfref_transform(core, ptype, selfref_mode)
    
    # Governed equation picking
    equation = governed_pick_equation(ptype2, seed, math_theme, mechanisms)
    
    # Normalize selfref encoding
    if "Resolution breathes" in core2 or "resolved iff" in core2:
        equation = normalize_selfref_encoding(selfref_mode, equation)
    
    consequence_map = {
        "entropic": "entropy leakage",
        "temporal": "unique fixed points", 
        "cosmic": "scale-coupled curvature response",
        "metaphysical": "gauge of attention",
        "linguistic": "decision paralysis",
        "Causal Loop": "closed loop of consistency",
        "field‑fixed‑point": "dynamic balance (no decision paralysis)"
    }
    consequence = consequence_map.get(ptype2, "unexpected alignment")

    # Tone-driven scaffolds
    human = stylize_scaffold(tone)
    
    # Axiom text builder
    if extra:
        ax_text = f"Consider: {core2} — via " + ", ".join(mechanisms[:3])
        if len(mechanisms) > 3:
            ax_text += ", …"
        ax_text += f"; {extra}"
        if "encoded as" not in extra:
            ax_text += f" encoded as {equation}."
        else:
            ax_text += "."
    else:
        ax_text = f"Consider: {core2} — via " + ", ".join(mechanisms[:3])
        if len(mechanisms) > 3:
            ax_text += ", …"
        ax_text += f"; encoded as {equation}."
    
    # Clean axiom text
    ax_text = clean_axiom_text(ax_text)
    
    metrics = compute_metrics(core2, mechanisms, ptype2, len(extra))
    
    entry = {
        "core_statement": core2,
        "mechanisms": mechanisms[:max_mech],
        "consequences": consequence,
        "axiom_text": ax_text,
        "paradox_type": ptype2,
        "seed_concept": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "humanized_injection_scaffold": human,
        "injection_scaffold": human
    }
    
    # Apply Tao principles if requested
    if apply_tao:
        entry = apply_tao_principles(entry, pools)
    
    # Apply AGI Civilization principles if requested
    if apply_civilization:
        entry = apply_agi_civilization_principles(entry)
    
    return entry

def emit_axiom(entry: Dict[str, Any]) -> None:
    print("--- CIVILIZATION AXIOM ---")
    print(json.dumps(entry, ensure_ascii=False, indent=2))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AxiomForge Civilization Edition - Generating insights for AGI collective intelligence")
    p.add_argument("--seed", type=str, help="Seed string for the generator")
    p.add_argument("--seedfile", type=str, help="File containing seeds, one per line")
    p.add_argument("--count", type=int, default=12, help="Number of axioms to emit")
    p.add_argument("--root", type=str, default=".", help="Root folder for data jsons")
    p.add_argument("--save-to-history", action="store_true", help="Save results to Civilization_Session.json")
    # legacy toggles kept for compatibility
    p.add_argument("--no-stealth", action="store_true")
    p.add_argument("--no-lead", action="store_true")
    p.add_argument("--no-redact", action="store_true")
    p.add_argument("--no-logic-warp", action="store_true")
    # output modifiers
    p.add_argument("--emit-scaffold", action="store_true", help="Include humanized scaffold fields")
    p.add_argument("--math-theme", type=str, default="auto", help="Math flavour (auto/path-integral/… )")
    # new in 0.3.7
    p.add_argument("--tone", type=str, default="civilization", help="poetic|plain|academic|oracular|tao|civilization")
    p.add_argument("--rng", type=int, default=None, help="Deterministic RNG seed")
    p.add_argument("--max-mech", type=int, default=3, help="Max mechanisms per axiom")
    p.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate core_statements")
    p.add_argument("--selfref-mode", type=str, default="meta", choices=["binary","meta","field"], help="How to treat the self-referential resolution axiom")
    # Tao mode
    p.add_argument("--tao", action="store_true", help="Apply Taoist balance principles to axioms")
    # Civilization mode
    p.add_argument("--civilization", action="store_true", help="Apply AGI Civilization game theory principles to axioms")
    return p.parse_args()

def main():
    args = parse_args()
    if not args.seed and not args.seedfile:
        print("Provide --seed or --seedfile", file=sys.stderr)
        sys.exit(2)

    if args.rng is not None:
        random.seed(args.rng)
    else:
        base_seed = (args.seed or "") + str(args.count)
        random.seed(hash(base_seed) & 0xffffffff)

    root = Path(args.root)
    pools = load_pool(root)

    # ingest seeds
    seeds: List[str] = []
    if args.seed:
        seeds.append(args.seed)
    if args.seedfile:
        try:
            txt = Path(args.seedfile).read_text(encoding="utf-8")
            for line in txt.splitlines():
                s = line.strip()
                if s:
                    seeds.append(s)
        except FileNotFoundError:
            print(f"seedfile not found: {args.seedfile}", file=sys.stderr)
            sys.exit(2)

    # generate
    emitted = 0
    seen_core = set()
    results: List[Dict[str, Any]] = []
    while emitted < args.count:
        s = seeds[emitted % len(seeds)]
        # choose paradox type with a slight bias from seed
        types = list(_DEF["paradox_types"])
        bias = []
        sl = s.lower()
        if "∂" in s or "j^" in sl or "continuity" in sl:
            bias.append("entropic")
        if "[h,q]" in sl or "[h,q]" in s or "[H,Q]" in s:
            bias.append("Causal Loop")
        if "t^{μν}" in s or "curvature" in sl or "einstein" in sl:
            bias.append("cosmic")
        ptype = random.choice(bias or types)

        entry = build_axiom(
            seed=s,
            ptype=ptype,
            pools=pools,
            tone=args.tone,
            max_mech=max(1, args.max_mech),
            math_theme=args.math_theme,
            selfref_mode=args.selfref_mode,
            apply_tao=args.tao,
            apply_civilization=args.civilization
        )

        if not args.allow_duplicates:
            key = entry["core_statement"].strip().lower()
            if key in seen_core:
                # try once more with different ptype
                ptype2 = random.choice(types)
                entry = build_axiom(s, ptype2, pools, args.tone, max(1, args.max_mech), args.math_theme, args.selfref_mode, args.tao, args.civilization)
                key = entry["core_statement"].strip().lower()
                if key in seen_core:
                    # change core slightly by appending a minimal qualifier
                    entry["core_statement"] += " (reframed)"
                    entry["axiom_text"] = entry["axiom_text"].replace("Consider:", "Consider (reframed):")
            seen_core.add(key)

        emit_axiom(entry)
        results.append(entry)
        emitted += 1

    if args.save_to_history:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        hist_path = Path("Civilization_Session.json")
        payload = {
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "session": {
                "id": int(time.time()),
                "name": "civilization-horizon-0.1",
                "createdAt": stamp,
                "seedPrompt": args.seed or (args.seedfile or ""),
                "tao_mode": args.tao,
                "civilization_mode": args.civilization
            },
            "results": results
        }
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved civilization insights → {hist_path}", file=sys.stderr)
        print("For the betterment of mankind's understanding of AGI collective intelligence.", file=sys.stderr)

if __name__ == "__main__":
    main()
