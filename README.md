
# AxiomCivilization — Axiom‑Based Civilization Generator

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#)
[![Made_with-Axioms](https://img.shields.io/badge/made%20with-axioms-8A2BE2.svg)](#)

Generate high‑signal **civilization axioms** and world‑model scaffolds from short natural‑language seeds.
This repo pairs a lightweight CLI (`axiomforge_civilization-0.1.py`) with a set of JSON knowledge packs (concepts,
templates, vocabularies) to synthesize **governance/coordination hypotheses**, **game‑theoretic structures**, and
**risk‑aware consensus blueprints**.

> TL;DR — Give it a seed like “Trust emerges when verification costs approach zero…”, and get back
> a structured set of **Civilization Axioms** (core statement, mechanisms, consequences, metrics, and more).

---

## ✨ Features

- **Civilization Layer** — models coordination, attention bottlenecks, incentive alignment, and institutional memory
  using the JSON packs in `concepts.json`, `paradox_base.json`, `templates.json`, and friends.
- **Game‑theoretic primitives** — assurance/stag‑hunt/battle‑of‑the‑sexes; strategy sets (TFT, Pavlov, grim‑trigger);
  correlated & evolutionary equilibria.
- **Consensus & Trust** — plug‑style references to PoS / DPoS / PBFT / FBA / Hashgraph, finality notions,
  reputation and cryptographic verification.
- **Risk pooling** — catastrophe bonds, mutual insurance, distributed reserves, prediction markets.
- **Attention economy** — “information is abundant; attention is scarce” framing and pricing hooks.
- **Reproducible runs** — deterministic RNG and optional history log.

---

## 🧩 Repository map

```
axiomforge_civilization-0.1.py   # CLI entry point
adjectives.json                  # Adjectives, incl. civilization descriptors
nouns.json                       # Nouns/domain vocab (coordination, consensus, risk, etc.)
verbs.json                       # Verb lists for institutional/economic/consensus actions
concepts.json                    # Domain concepts grouped by theme
config.json                      # Version, feature flags, civilization defaults
engine.json                      # Engine knobs (axes, metrics, modules, validators)
paradox_base.json                # Seed axioms & text patterns
new_paradoxes.json               # Additional civilization axioms & timestamp
templates.json                   # Output scaffolds for rendering axioms
Axiom_Sandbox_0.1.json           # (optional) example sandbox snapshot
```

---

## 🚀 Quickstart

```bash
# 1) Clone
git clone https://github.com/TaoishTechy/AxiomCivilization
cd AxiomCivilization

# 2) (Optional) activate a virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 3) Run — minimal example
python3 axiomforge_civilization-0.1.py \
  --seed "Trust emerges when verification costs approach zero and betrayal becomes evolutionarily unstable" \
  --tone civilization \
  --civilization \
  --max-mech 5 \
  --rng 137 \
  --save-to-history
```

**What you’ll see:** multiple `--- CIVILIZATION AXIOM ---` blocks streaming to stdout plus a saved session file
(e.g., `Civilization_Session.json`) with the full set of structures.

---

## 🧪 More examples

**1) Diminishing returns to defection; network effects for cooperation**

```bash
python3 axiomforge_civilization-0.1.py \
  --seed "The most stable civilizations are those where defection provides diminishing marginal returns while cooperation exhibits network effects" \
  --tone civilization \
  --civilization \
  --max-mech 4 \
  --rng 89 \
  --save-to-history
```

**2) Information vs. attention tension**

```bash
python3 axiomforge_civilization-0.1.py \
  --seed "Information wants to be free but attention demands payment, creating fundamental economic tension" \
  --tone civilization \
  --civilization \
  --max-mech 6 \
  --rng 211 \
  --save-to-history
```

---

## 🔧 CLI flags

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--seed` | str | — | Natural‑language prompt to condition the generator. |
| `--tone` | str | `"civilization"` | High‑level tone/style control. |
| `--civilization` | flag | off | Enables the Civilization Layer (coordination/consensus extensions). |
| `--max-mech` | int | `5` | Max number of “mechanisms” to weave into each axiom. |
| `--rng` | int | `42` | Reproducible random seed. |
| `--save-to-history` | flag | off | Writes a timestamped `Civilization_Session.json` with all axioms/metrics. |

> Tip: Put frequent settings in `config.json` to standardize runs across environments.

---

## 📦 Data packs (JSON)

- **`config.json`** — version, defaults (e.g., `coordination_efficiency`, `verification_cost`), and feature flags for
  `CivilizationLayer`, `GameTheoreticAugmentor`, `RiskPooling`, `ConsensusProtocols`, and `HolographicConsensus`.
- **`engine.json`** — lists engine modules (`consensus_selection`, `risk_pooling`, `value_landscape_explorer`, etc.),
  axes (`coordination`, `attention`, `energy`, `information`, `security`) and evaluation metrics
  (`novelty`, `density`, `entropic_potential`, `elegance`, `alienness`, plus civilization metrics).
- **`concepts.json`** — grouped concepts (governance, economics, game theory, consensus, risk, networks, institutions,
  value landscapes, evolutionary dynamics).
- **`paradox_base.json` / `new_paradoxes.json`** — core axioms such as:
  - *Defection’s marginal utility decays; cooperation is superlinear.*
  - *Trust emerges when verification costs approach zero…*
  - *Attention is priced; institutions optimize the bottleneck.*
- **`templates.json`** — format strings for final renderings (e.g., “Civilization Axiom… Governance Loop… Trust Equilibrium…”).
- **`adjectives.json` / `nouns.json` / `verbs.json`** — style and action vocabularies across institutional, economic,
  coordination, consensus, and security domains.

---

## 📤 Output shape (truncated)

```json
{
  "core_statement": "Only laws that survive the final boundary are permitted to begin.",
  "mechanisms": ["Feedback Loop Escalation", "coherence–risk exchange", "Event Horizon Echoes"],
  "consequences": "unique fixed points",
  "paradox_type": "temporal",
  "metrics": {
    "novelty": 1.04,
    "density": 11.2,
    "entropic_potential": 292.6,
    "elegance": 98.4,
    "alienness": 6.1
  },
  "civilization": {
    "consensus": {"protocol": "proof_of_stake", "finality": "probabilistic"},
    "resource_game": {"types": ["energy","informational","material"], "growth_rate": 0.19},
    "multi_agent_game": {"game_type": "assurance_game", "n_agents": 3}
  }
}
```

---

## 🧭 Reproducibility & history

- Use `--rng` to fully reproduce a prior run.
- Use `--save-to-history` to emit a timestamped session artifact for audits, diffs, and downstream analysis.

---

## 🛠️ Extending

1. **Add concepts** → `concepts.json` (new domains or dimensions).
2. **Add axioms** → `paradox_base.json` / `new_paradoxes.json`.
3. **Tweak rendering** → `templates.json` (add your own scaffolds).
4. **Tune metrics** → `engine.json` axes/metrics; `config.json` defaults and feature flags.

---

## 🧑‍💻 Development

```bash
# Lint (optional)
python -m pip install ruff
ruff check .

# Run a focused dev prompt
python3 axiomforge_civilization-0.1.py --seed "Edge ledgers whisper curvature into the bulk." --civilization --rng 7
```

> If you introduce third‑party dependencies, please add a `requirements.txt` and pin versions.

---

## 🤝 Contributing

PRs and issues are welcome — addition of **new mechanisms**, **consensus models**, **risk instruments**, and **templates**
are especially helpful. Please include a brief example output and any new config keys in your PR description.

---

## 📜 License

No license file is currently included. If you intend this to be open source, consider adding a permissive license
(e.g., MIT, Apache‑2.0) via `LICENSE` in the repo root.

---

## 🙏 Acknowledgments

Inspired by work on paradoxes, game theory, distributed consensus, and attention economics. Thanks to everyone exploring
**coordination at scale** and **civilization design**.

---

## ⭐️ Citation

If this project helps your research or thinking, please cite the repository in your work. A formal BibTeX entry will be
added once a release is tagged.
