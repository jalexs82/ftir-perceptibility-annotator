# 🔬 FTIR Feature Perceptibility & Discriminatory Power Annotator

An interactive psychophysical & forward-electrodynamic web application for evaluating the human and neural-model perceptibility of diagnostic microplastic FTIR spectral features under calibrated environmental distortions ($4000\text{--}1200\text{ cm}^{-1}$).

---

## 🚀 Quick Start (Run Anywhere)

This repository is **100% self-contained** — all reference polymer templates (11 polymer classes), authentic riverbed/environmental backgrounds (`data/reference_bundle.npz`), calibrated parameter distributions (`data/environmental_cohort_distortion_parameters.yaml`), pre-trained neural model weights (`data/ftir_perceptibility_model.pt`), and saved annotations (`data/perceptibility_trials.jsonl`) are bundled directly inside `data/`.

### 1. Install Dependencies
```bash
pip install numpy scipy pandas pyyaml torch
```

### 2. Launch the Annotator
```bash
python run_app.py
```
Then open **`http://127.0.0.1:8055`** in your browser.

---

## 🎯 Key Features & Organization

1. **Polymer Group $\rightarrow$ Diagnostic Peak Hierarchy:**
   - Organized by Polymer Family (`Polyolefins`, `Polyesters/Acrylates/Carbonates`, `Styrenics & Nitriles`, `Polyamides & Polyurethanes`, `Chlorinated Polymers`, and `Spurious Peak & False-Positive Tests`).
   - Selecting a diagnostic peak or peak combination automatically draws a relevant polymer spectrum from that group and displays a clear explanation of **what the feature distinguishes** (e.g., ABS vs. PS, PE vs. PP, PET vs. PLA/PC).
2. **Yellow Target Highlight Corridors:**
   - The current peak or combination of interest is highlighted in **bright yellow (`#facc15`)** on both the full mid-IR spectrum and the zoomed diagnostic window.
3. **Comprehensive Environmental Degradation & Failure Modes:**
   - `compound`: Coupled optical attenuation ($\tau$), Kramers–Kronig anomalous dispersion Mie scattering ($\rho_0$), pink $1/f$ detector noise, baseline slant, and subtle hydration/subtraction mismatch.
   - `under_subtracted`: Positive bleed-through of natural silicate, carbonate, and clay backgrounds.
   - `over_subtracted`: Negative baseline troughs from over-subtracted annular backgrounds.
   - `peak_saturation`: Thick-particle Beer–Lambert / detector saturation where peaks flatten into plateaus and troughs fill in.
   - `biofilm_hydration`: Broad O–H ($3350\text{ cm}^{-1}$), H–O–H bend ($1640\text{ cm}^{-1}$), and microbial EPS overlap.
   - `spurious_intrusion`: Tests **false-positive peak appearance** (e.g. biofilm Amide I/II growing on polyolefins, photo-oxidative carbonyl on PE/PP/PS, or mineral calcite at $1425\text{ cm}^{-1}$).
4. **Live Tally & Review Modal:**
   - Click **"📋 Review Trials & Tallies"** in the top bar to inspect all recorded trials, view per-feature tallies, and delete any accidental entries.

---

## 💾 Synchronizing Annotations via GitHub

All recorded trials are saved to `data/perceptibility_trials.jsonl`. Collaborators can pull the repo, run annotations, and commit/push `data/perceptibility_trials.jsonl` back to merge their trials.
