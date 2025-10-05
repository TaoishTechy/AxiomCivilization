# Axiom Civilization Simulator — GUI

This Streamlit app visualizes and *simulates* the civilizations produced by `axiomforge_civilization-0.1.py`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run axiom-gui.py
```

Then in the sidebar:
1. Set **Path to generator script** (e.g. `axiomforge_civilization-0.1.py`).
2. Paste or edit a **Seed**.
3. Click **Generate Now** or **Load Civilization_Session.json**.

## What it does

- Parses any CLI output that contains blocks like:

```
--- CIVILIZATION AXIOM ---
{ ...json... }
```

- Supports *recursive* seed evolution and *real-time* visualization (oscilloscopes & radar).

## 12 Enhancements

1. **QRSE Seeds** – quantum-recursive seed evolver.
2. **AGRO Oscilloscope** – attention gauge oscillator (real-time).
3. **HFN Frontier** – holographic Pareto navigator.
4. **TBC Trust Basin** – value-landscape potential cartography.
5. **CPS Percolation** – coordination percolation sweep.
6. **CFA Consensus** – finality fault avalanche Monte Carlo.
7. **PDT Paradox** – paradox stability score.
8. **ADWR War Room** – attack–defense coverage matrix.
9. **IDO Drift** – institutional stage transition oracle.
10. **MEW Entanglement** – mechanism co-occurrence network.
11. **RPSL Risk** – risk-pool stress VaR.
12. **CARR Radar** – coevolutionary arms-race radar (real-time).

## Tips

- If your generator prints a lot, open the “Generator Output” expander **after** a run.
- Auto-refresh can be toggled in the sidebar for time-based widgets.
- The app is safe with missing fields; unknown items are defaulted.
