"""
perceptibility_app/engine.py
============================
Backend Physical Synthesis & Model Inference Engine for the Interactive Perceptibility App:
- Incorporates calibrated parameters from assets/environmental_cohort_distortion_parameters.yaml
- Supports comprehensive diagnostic feature taxonomy (18 diagnostic features organized by Polymer Family)
- Simulates true electrodynamic forward physical degradation & failure modes:
  1. Beer-Lambert Optical Saturation / Attenuation (tau)
  2. Resonant Kramers-Kronig Anomalous Dispersion Mie Scattering (rho_0, Q_ext) + Peak Center Shifts
  3. Real Empirical Annular Background Coupling (BSR, SCR, Silt/Clay/Carbonate)
  4. Active-Mask Pink (1/f) Instrument Noise (MAD-calibrated SNR)
  5. Annular Under/Over-Subtraction & Zero-Clipping
  6. Peak Saturation / Detector Dynamic Limit Clipping (Plateau Flattening)
  7. Biofilm & Hydration Overlap (Water OH/bend + Microbial EPS)
  8. Spurious Peak Intrusion (False-positive diagnostic band mimicry: e.g. Biofilm Amide on PE/PP, Weathering Carbonyl on Polyolefins, Carbonate on Aromatic/CH2, CO2/Rovibrational spikes)
"""

import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import torch
import torch.nn as nn

# Ensure autoLUMOS root is on path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
BUNDLE_PATH = os.path.join(DATA_DIR, "reference_bundle.npz")

WN_GRID = np.arange(4000.0, 1196.0 - 1e-4, -4.0)
N_CHANNELS = len(WN_GRID)

# Legacy 7 features for the pre-trained neural model compatibility
MODEL_FEATURE_NAMES = [
    'PE_doublet',
    'PP_cluster',
    'PP_umbrella',
    'Carbonyl_peak',
    'Amide_II',
    'Aromatic_ring',
    'Nitrile_spike'
]
NUM_MODEL_FEATURES = len(MODEL_FEATURE_NAMES)

# Comprehensive 18-Feature Taxonomy mapped to Polymer Families and Candidate Polymers
COMPREHENSIVE_FEATURES = {
    # 1. Polyolefins (PE, PP)
    'PE_CH2_doublet': {
        'family': 'Polyolefins (PE & PP)',
        'title': 'PE Aliphatic CH₂ Asymmetric/Symmetric Doublet (2916 & 2848 cm⁻¹)',
        'polymers': ['PE'],
        'confounders': ['PP', 'Nylon', 'PVC'],
        'distinguishes': 'Distinguishes Polyethylene (PE) from Polypropylene (PP) and other aliphatic polymers via the clean 2-peak split without a 2950 cm⁻¹ methyl shoulder.',
        'diagnostic_question': 'Can you clearly resolve the twin CH₂ peaks (2916 & 2848 cm⁻¹) and the central splitting trough at 2880 cm⁻¹?',
        'visual_cues': [
            'Inspect the yellow-highlighted 2916 and 2848 cm⁻¹ twin peaks and the 2880 cm⁻¹ dip.',
            'In peak saturation mode, watch when the 2880 cm⁻¹ trough fills in into a flat-topped plateau.',
            'Confirm absence of a 2950 cm⁻¹ CH₃ shoulder (which would indicate PP).'
        ],
        'window': [2750.0, 3050.0],
        'target_ranges': [[2900.0, 2932.0], [2835.0, 2862.0]],
        'primary_wn': 2916.0,
        'secondary_wn': 2848.0,
        'model_proxy': 'PE_doublet'
    },
    'PE_CH2_scissoring': {
        'family': 'Polyolefins (PE & PP)',
        'title': 'PE Crystalline CH₂ Scissoring Band (1464 / 1472 cm⁻¹)',
        'polymers': ['PE'],
        'confounders': ['PP', 'PVC'],
        'distinguishes': 'Distinguishes Polyethylene (PE) backbone methylene bending (1464 cm⁻¹) and confirms absence of the PP 1377 cm⁻¹ methyl umbrella.',
        'diagnostic_question': 'Is the sharp 1464 cm⁻¹ CH₂ scissoring peak clearly perceptible above mineral carbonate/baseline interference?',
        'visual_cues': [
            'Look at the yellow corridor around 1464 cm⁻¹ for a sharp methylene deformation spike.',
            'Check 1377 cm⁻¹ (red negative marker): pure HDPE has no peak there, whereas PP has a strong 1377 cm⁻¹ peak.',
            'Watch out for calcium carbonate (CaCO₃) mineral bleed around 1420–1450 cm⁻¹.'
        ],
        'window': [1320.0, 1540.0],
        'target_ranges': [[1450.0, 1480.0]],
        'primary_wn': 1464.0,
        'secondary_wn': 1377.0,
        'model_proxy': 'PE_doublet'
    },
    'PP_CH3_CH2_cluster': {
        'family': 'Polyolefins (PE & PP)',
        'title': 'PP 4-Peak Methyl/Methylene C–H Cluster (2950, 2918, 2873, 2838 cm⁻¹)',
        'polymers': ['PP'],
        'confounders': ['PE', 'PS', 'PMMA'],
        'distinguishes': 'Distinguishes Polypropylene (PP) from Polyethylene (PE) via the characteristic 4-peak complex led by the prominent 2950 cm⁻¹ asymmetric CH₃ stretch.',
        'diagnostic_question': 'Can you distinguish the multi-peak PP cluster (especially the 2950 cm⁻¹ CH₃ peak separated from 2918 cm⁻¹) rather than a simple 2-peak doublet?',
        'visual_cues': [
            'Focus on the yellow corridor: 2950 cm⁻¹ (CH₃) + 2918 cm⁻¹ (CH₂) + 2873/2838 cm⁻¹.',
            'Breakdown occurs when noise or Mie scattering merges 2950 and 2918 cm⁻¹ into an indistinct lump.',
            'Use the 2nd derivative (purple dashed) to see if the 4 inflection minima survive.'
        ],
        'window': [2750.0, 3050.0],
        'target_ranges': [[2938.0, 2968.0], [2905.0, 2930.0], [2860.0, 2885.0]],
        'primary_wn': 2950.0,
        'secondary_wn': 2918.0,
        'model_proxy': 'PP_cluster'
    },
    'PP_umbrella_1377': {
        'family': 'Polyolefins (PE & PP)',
        'title': 'PP Symmetric CH₃ Umbrella Deformation (1377 cm⁻¹ vs 1456 cm⁻¹)',
        'polymers': ['PP'],
        'confounders': ['PE', 'PLA', 'PMMA'],
        'distinguishes': 'Definitively confirms Polypropylene (PP) vs Polyethylene (PE). PP exhibits a strong, sharp 1377 cm⁻¹ symmetric methyl umbrella alongside the 1456 cm⁻¹ band.',
        'diagnostic_question': 'Is the 1377 cm⁻¹ symmetric CH₃ umbrella peak clearly discernible alongside the 1456 cm⁻¹ bending peak?',
        'visual_cues': [
            'Inspect the yellow-highlighted 1377 cm⁻¹ peak and compare its height to the 1456 cm⁻¹ companion.',
            'In PE, 1377 cm⁻¹ is absent or tiny; in PP, it is a crisp, unmistakable diagnostic spike.',
            'Watch how under-subtracted silicate/carbonate baselines tilt or obscure the 1350–1450 cm⁻¹ region.'
        ],
        'window': [1300.0, 1520.0],
        'target_ranges': [[1362.0, 1392.0]],
        'primary_wn': 1377.0,
        'secondary_wn': 1456.0,
        'model_proxy': 'PP_umbrella'
    },

    # 2. Polyesters, Acrylates & Carbonates (PET, PLA, PMMA, PC)
    'PET_conjugated_ester': {
        'family': 'Polyesters, Acrylates & Carbonates',
        'title': 'PET Conjugated Ester Carbonyl (1715 cm⁻¹) & C–O–C (1245 cm⁻¹)',
        'polymers': ['PET'],
        'confounders': ['PLA', 'PMMA', 'PC', 'PU'],
        'distinguishes': 'Distinguishes Aromatic Polyester (PET, 1715 cm⁻¹) from Aliphatic Polyester (PLA, 1750 cm⁻¹), Acrylate (PMMA, 1730 cm⁻¹), and Polycarbonate (PC, 1775 cm⁻¹).',
        'diagnostic_question': 'Can you clearly perceive the sharp 1715 cm⁻¹ ester carbonyl peak and verify its position near ~1715 cm⁻¹?',
        'visual_cues': [
            'Inspect the yellow-highlighted 1715 cm⁻¹ carbonyl peak.',
            'Notice how anomalous dispersion Mie scattering can shift the apparent peak maximum by 10–15 cm⁻¹ or create a deep derivative trough at 1760 cm⁻¹.',
            'Confirm companion ester C–O stretch at 1245 cm⁻¹ and terephthalate ring at 1410/1505 cm⁻¹.'
        ],
        'window': [1620.0, 1840.0],
        'target_ranges': [[1695.0, 1732.0]],
        'primary_wn': 1715.0,
        'secondary_wn': 1245.0,
        'model_proxy': 'Carbonyl_peak'
    },
    'PLA_aliphatic_ester': {
        'family': 'Polyesters, Acrylates & Carbonates',
        'title': 'PLA High-Frequency Aliphatic Ester C=O (1750 cm⁻¹)',
        'polymers': ['PLA'],
        'confounders': ['PET', 'PMMA', 'PC'],
        'distinguishes': 'Distinguishes Polylactic Acid (PLA, ~1750 cm⁻¹) from PET (1715 cm⁻¹) and PMMA (1730 cm⁻¹), paired with absence of aromatic ring peaks.',
        'diagnostic_question': 'Is the high-frequency 1750 cm⁻¹ aliphatic ester C=O peak clearly distinguishable above baseline and scattering distortion?',
        'visual_cues': [
            'Look at the yellow corridor centered at 1750 cm⁻¹.',
            'Compare against 1715 cm⁻¹ (PET) and 1775 cm⁻¹ (PC): peak center stability is critical here.',
            'Check 1452 cm⁻¹ (CH₃ bend) and 1360 cm⁻¹ (CH bend) in the full spectrum.'
        ],
        'window': [1640.0, 1850.0],
        'target_ranges': [[1732.0, 1768.0]],
        'primary_wn': 1750.0,
        'secondary_wn': 1452.0,
        'model_proxy': 'Carbonyl_peak'
    },
    'PMMA_acrylate_ester': {
        'family': 'Polyesters, Acrylates & Carbonates',
        'title': 'PMMA Methacrylate Ester C=O (1730 cm⁻¹) & CH₃ Bend (1450 cm⁻¹)',
        'polymers': ['PMMA'],
        'confounders': ['PET', 'PLA', 'PU'],
        'distinguishes': 'Distinguishes Acrylic / PMMA (1730 cm⁻¹ ester C=O + 1450 cm⁻¹ α-methyl bend, no aromatic rings, no N–H amide bands) from PET, PLA, and PU.',
        'diagnostic_question': 'Can you clearly identify the 1730 cm⁻¹ methacrylate carbonyl spike and distinguish it from surrounding interference?',
        'visual_cues': [
            'Inspect the yellow corridor at 1730 cm⁻¹.',
            'Watch how water H–O–H bending (1640 cm⁻¹) or biofilm Amide I (1650 cm⁻¹) can encroach on the low-wavenumber side.',
            'Verify absence of Amide II (1540 cm⁻¹) which differentiates PMMA from Polyurethane (PU).'
        ],
        'window': [1620.0, 1840.0],
        'target_ranges': [[1712.0, 1748.0]],
        'primary_wn': 1730.0,
        'secondary_wn': 1450.0,
        'model_proxy': 'Carbonyl_peak'
    },
    'PC_carbonate_1775': {
        'family': 'Polyesters, Acrylates & Carbonates',
        'title': 'PC High-Frequency Carbonate C=O (1775 cm⁻¹) & Bisphenol Ring (1504 cm⁻¹)',
        'polymers': ['PC'],
        'confounders': ['PET', 'PLA', 'PS'],
        'distinguishes': 'Uniquely identifies Polycarbonate (PC) via its unusually high 1775 cm⁻¹ carbonate carbonyl paired with the sharp 1504 cm⁻¹ Bisphenol-A aromatic ring.',
        'diagnostic_question': 'Is the diagnostic 1775 cm⁻¹ carbonate C=O peak clearly perceptible and distinct from normal ester carbonyls?',
        'visual_cues': [
            'Focus on the yellow corridor at 1775 cm⁻¹—the highest carbonyl frequency among common microplastics.',
            ' Also check the sharp 1504 cm⁻¹ Bisphenol-A ring spike in the full spectrum.',
            'In thick particles, check if saturation broadens 1775 cm⁻¹ downward toward 1740 cm⁻¹.'
        ],
        'window': [1450.0, 1850.0],
        'target_ranges': [[1755.0, 1795.0], [1490.0, 1518.0]],
        'primary_wn': 1775.0,
        'secondary_wn': 1504.0,
        'model_proxy': 'Carbonyl_peak'
    },
    'Ester_CO_backbone': {
        'family': 'Polyesters, Acrylates & Carbonates',
        'title': 'Ester / Carbonate Asymmetric C–O–C Stretch (1210–1260 cm⁻¹)',
        'polymers': ['PET', 'PC', 'PMMA', 'PLA', 'PU'],
        'confounders': ['PE', 'PP', 'PS'],
        'distinguishes': 'Confirms oxygen-containing ester/carbonate/urethane backbones (intense C–O–C band at 1210–1260 cm⁻¹) vs oxidized polyolefins (which have weathered C=O but weak C–O).',
        'diagnostic_question': 'Can you clearly perceive the intense C–O–C ester stretching band around 1210–1260 cm⁻¹ near the low-wavenumber cutoff?',
        'visual_cues': [
            'Look at the yellow corridor between 1210 and 1265 cm⁻¹.',
            'This region sits directly adjacent to the intense silicate Si–O mineral tail (1000–1150 cm⁻¹).',
            'Under-subtracted sediment backgrounds heavily distort or swallow this band first.'
        ],
        'window': [1196.0, 1450.0],
        'target_ranges': [[1210.0, 1265.0]],
        'primary_wn': 1245.0,
        'secondary_wn': 1715.0,
        'model_proxy': 'Carbonyl_peak'
    },

    # 3. Styrenics & Aromatics (PS, ABS, PET, PC)
    'Styrenic_aromatic_duet': {
        'family': 'Styrenics & Nitriles (PS & ABS)',
        'title': 'Styrenic Aromatic Ring Quadrant & Semicircle Duet (1601 & 1493 cm⁻¹)',
        'polymers': ['PS', 'ABS'],
        'confounders': ['PE', 'PP', 'PVC'],
        'distinguishes': 'Identifies mono-substituted phenyl rings in Polystyrene (PS) and ABS via the sharp 1601 cm⁻¹ and 1493 cm⁻¹ pair alongside 1452 cm⁻¹.',
        'diagnostic_question': 'Can you clearly resolve both sharp aromatic ring peaks at 1601 cm⁻¹ and 1493 cm⁻¹ above baseline/biofilm interference?',
        'visual_cues': [
            'Inspect the two yellow-highlighted corridors at 1601 cm⁻¹ and 1493 cm⁻¹.',
            '1601 cm⁻¹ is vulnerable to overlap from water bending (1640 cm⁻¹) and biofilm Amide I/II (1650/1545 cm⁻¹).',
            '1493 cm⁻¹ and 1452 cm⁻¹ form a characteristic close doublet in PS and ABS.'
        ],
        'window': [1400.0, 1680.0],
        'target_ranges': [[1585.0, 1615.0], [1478.0, 1508.0]],
        'primary_wn': 1601.0,
        'secondary_wn': 1493.0,
        'model_proxy': 'Aromatic_ring'
    },
    'Aromatic_CH_3026': {
        'family': 'Styrenics & Nitriles (PS & ABS)',
        'title': 'Unsaturated Aromatic =C–H Stretch (>3000 cm⁻¹: 3026 & 3060 cm⁻¹)',
        'polymers': ['PS', 'ABS'],
        'confounders': ['PE', 'PP', 'PMMA'],
        'distinguishes': 'Confirms unsaturated aromatic C–H stretching immediately to the left (>3000 cm⁻¹) of the saturated aliphatic C–H envelope (<3000 cm⁻¹).',
        'diagnostic_question': 'Can you perceive the aromatic =C–H peak cluster at 3026–3060 cm⁻¹ on the high-wavenumber side of the 2920 cm⁻¹ aliphatic band?',
        'visual_cues': [
            'Focus on the yellow corridor at 3010–3075 cm⁻¹ just left of the 3000 cm⁻¹ boundary.',
            'In hydration/biofilm mode, the broad O–H band (3350 cm⁻¹) sweeps across 3026 cm⁻¹ and can mask this subtle cluster.',
            'Check the 2nd derivative for a sharp negative trough right at 3026 cm⁻¹.'
        ],
        'window': [2800.0, 3200.0],
        'target_ranges': [[3010.0, 3075.0]],
        'primary_wn': 3026.0,
        'secondary_wn': 2924.0,
        'model_proxy': 'Aromatic_ring'
    },
    'ABS_nitrile_2237': {
        'family': 'Styrenics & Nitriles (PS & ABS)',
        'title': 'ABS Exclusive Nitrile C≡N Stretch Spike (2237 cm⁻¹)',
        'polymers': ['ABS'],
        'confounders': ['PS'],
        'distinguishes': 'The single decisive discriminator separating Acrylonitrile Butadiene Styrene (ABS) from pure Polystyrene (PS). Located in the silent mid-IR window (2000–2500 cm⁻¹).',
        'diagnostic_question': 'Can you detect the isolated 2237 cm⁻¹ nitrile C≡N spike standing clearly above the mid-IR noise floor?',
        'visual_cues': [
            'Inspect the yellow corridor at 2237 cm⁻¹.',
            'Almost no other polymer or mineral has a fundamental vibration at 2237 cm⁻¹.',
            'Distinguish it from atmospheric CO₂ (2349 cm⁻¹ doublet) if spurious gas interference is present.'
        ],
        'window': [2100.0, 2420.0],
        'target_ranges': [[2218.0, 2255.0]],
        'primary_wn': 2237.0,
        'secondary_wn': 1601.0,
        'model_proxy': 'Nitrile_spike'
    },

    # 4. Polyamides & Polyurethanes (Nylon, PU)
    'Amide_I_II_duet': {
        'family': 'Polyamides & Polyurethanes (Nylon & PU)',
        'title': 'Polyamide Amide I (1640 cm⁻¹) & Amide II (1540 cm⁻¹) Twin Pair',
        'polymers': ['Nylon'],
        'confounders': ['PE', 'PP', 'PU'],
        'distinguishes': 'Identifies Polyamide (Nylon) via the intense, balanced Amide I (1640 cm⁻¹ C=O) and Amide II (1540 cm⁻¹ N–H/C–N) doublet.',
        'diagnostic_question': 'Are both the Amide I (~1640 cm⁻¹) and Amide II (~1540 cm⁻¹) peaks clearly perceptible as a paired doublet?',
        'visual_cues': [
            'Inspect the two yellow corridors at 1640 cm⁻¹ and 1540 cm⁻¹.',
            'Hydration (1640 cm⁻¹ H–O–H bend) only inflates the 1640 cm⁻¹ band; true Nylon requires the companion 1540 cm⁻¹ Amide II peak.',
            'Warning: Heavy proteinaceous biofilm can mimic weak Amide I/II on weathered polyolefins!'
        ],
        'window': [1450.0, 1750.0],
        'target_ranges': [[1620.0, 1660.0], [1520.0, 1562.0]],
        'primary_wn': 1540.0,
        'secondary_wn': 1640.0,
        'model_proxy': 'Amide_II'
    },
    'PU_urethane_triplet': {
        'family': 'Polyamides & Polyurethanes (Nylon & PU)',
        'title': 'Polyurethane Carbonyl (1725 cm⁻¹) + Amide II (1530 cm⁻¹) Combination',
        'polymers': ['PU'],
        'confounders': ['Nylon', 'PET', 'PMMA'],
        'distinguishes': 'Distinguishes Polyurethane (PU) from Nylon (which lacks the 1725 cm⁻¹ ester/urethane C=O) and from Polyesters/PMMA (which lack the 1530 cm⁻¹ N–H/C–N Amide II peak).',
        'diagnostic_question': 'Can you perceive BOTH the ~1725 cm⁻¹ urethane C=O band AND the ~1530 cm⁻¹ Amide II band that together define PU?',
        'visual_cues': [
            'Inspect both yellow-highlighted corridors: 1725 cm⁻¹ (Urethane C=O) and 1530 cm⁻¹ (Urethane Amide II).',
            'If 1530 cm⁻¹ disappears into noise, PU can be misclassified as an acrylate/polyester (PMMA/PET).',
            'If 1725 cm⁻¹ is obscured, PU can be confused with Nylon.'
        ],
        'window': [1450.0, 1820.0],
        'target_ranges': [[1700.0, 1748.0], [1510.0, 1555.0]],
        'primary_wn': 1725.0,
        'secondary_wn': 1530.0,
        'model_proxy': 'Amide_II'
    },
    'Amide_A_NH_3300': {
        'family': 'Polyamides & Polyurethanes (Nylon & PU)',
        'title': 'Hydrogen-Bonded Amide A / Urethane N–H Stretch (3300–3320 cm⁻¹)',
        'polymers': ['Nylon', 'PU'],
        'confounders': ['PE', 'PP', 'PET'],
        'distinguishes': 'Confirms N–H stretching in Polyamides (Nylon, sharp peak at 3300 cm⁻¹) and Polyurethanes (PU, 3320 cm⁻¹) vs surface water O–H (broad dome at 3350–3400 cm⁻¹).',
        'diagnostic_question': 'Can you distinguish the structured N–H stretching peak (~3300 cm⁻¹) from broad background hydration or noise?',
        'visual_cues': [
            'Look at the yellow corridor around 3270–3340 cm⁻¹.',
            'True Nylon N–H is much narrower and more symmetric than liquid water O–H.',
            'Under heavy hydration/biofilm degradation, watch when the broad water dome completely swallows the discrete N–H peak.'
        ],
        'window': [3050.0, 3650.0],
        'target_ranges': [[3265.0, 3345.0]],
        'primary_wn': 3300.0,
        'secondary_wn': 2928.0,
        'model_proxy': 'Amide_II'
    },

    # 5. Chlorinated Polymers (PVC)
    'PVC_chlorinated_duet': {
        'family': 'Chlorinated Polymers (PVC)',
        'title': 'PVC Chlorinated CH₂–Cl Bend (1428 cm⁻¹) & CH–Cl Rocking (1255 cm⁻¹)',
        'polymers': ['PVC'],
        'confounders': ['PE', 'PP'],
        'distinguishes': 'Distinguishes Polyvinyl Chloride (PVC) from polyolefins via the shifted CH₂–Cl bending at 1428 cm⁻¹ (vs 1464 cm⁻¹ in PE) and characteristic CH–Cl rocking at 1255 cm⁻¹.',
        'diagnostic_question': 'Are the diagnostic 1428 cm⁻¹ (CH₂–Cl) and 1255 cm⁻¹ (CH–Cl) peaks clearly resolvable?',
        'visual_cues': [
            'Inspect the two yellow corridors at 1428 cm⁻¹ and 1255 cm⁻¹.',
            'Note how 1428 cm⁻¹ is shifted ~36 cm⁻¹ lower than normal PE CH₂ scissoring (1464 cm⁻¹) due to the electronegative chlorine atom.',
            'Watch out for mineral calcite/carbonate interference right at 1420 cm⁻¹!'
        ],
        'window': [1200.0, 1520.0],
        'target_ranges': [[1412.0, 1442.0], [1238.0, 1272.0]],
        'primary_wn': 1428.0,
        'secondary_wn': 1255.0,
        'model_proxy': 'PE_doublet'
    },

    # 6. Spurious Peak / False-Positive Interference Diagnostics
    'Spurious_Biofilm_Amide_on_Polyolefin': {
        'family': 'Spurious Peak & False-Positive Tests',
        'title': 'Spurious Biofilm Amide I/II Appearance on PE/PP (1650 & 1545 cm⁻¹)',
        'polymers': ['PE', 'PP'],
        'confounders': ['Nylon', 'PU'],
        'distinguishes': 'Tests FALSE-POSITIVE appearance: Microbial biofilm (proteins/EPS) growing on weathered PE or PP introduces spurious Amide I (1650 cm⁻¹) and Amide II (1545 cm⁻¹) peaks that mimic Nylon!',
        'diagnostic_question': 'At what point along t do the growing spurious biofilm peaks (1650 & 1545 cm⁻¹) become prominent enough to trick a classifier or analyst into seeing an Amide/Nylon signature?',
        'visual_cues': [
            'At t = 0.0, the yellow corridors (1650 & 1545 cm⁻¹) are completely flat (pure PE or PP).',
            'As you slide t toward 1.0, microbial protein Amide I/II bands grow into the yellow corridors.',
            'Mark the boundary t where the spurious amide peaks look like real polymer features!'
        ],
        'window': [1420.0, 1780.0],
        'target_ranges': [[1625.0, 1675.0], [1525.0, 1565.0]],
        'primary_wn': 1650.0,
        'secondary_wn': 1545.0,
        'is_spurious_test': True,
        'spurious_type': 'biofilm_amide',
        'model_proxy': 'Amide_II'
    },
    'Spurious_Weathering_Carbonyl_on_Polyolefin': {
        'family': 'Spurious Peak & False-Positive Tests',
        'title': 'Spurious Photo-Oxidation Carbonyl Appearance on PE/PP/PS (1715–1735 cm⁻¹)',
        'polymers': ['PE', 'PP', 'PS'],
        'confounders': ['PET', 'PLA', 'PMMA', 'PU'],
        'distinguishes': 'Tests FALSE-POSITIVE appearance: UV photo-oxidation and environmental ketone/acid weathering on PE, PP, or PS creates a spurious 1718 cm⁻¹ carbonyl peak mimicking an ester!',
        'diagnostic_question': 'At what point along t does the growing photo-oxidative carbonyl peak (~1718 cm⁻¹) become prominent enough to be perceived as a true functional carbonyl band?',
        'visual_cues': [
            'At t = 0.0, pure PE/PP/PS has NO carbonyl peak at 1718 cm⁻¹.',
            'As you increase t, a photo-oxidative ketone/lactone band grows at 1718 cm⁻¹ alongside ester-like C–O humps.',
            'Mark the cutoff t where you perceive a distinct, unmistakable carbonyl peak in the yellow window.'
        ],
        'window': [1600.0, 1850.0],
        'target_ranges': [[1700.0, 1745.0]],
        'primary_wn': 1718.0,
        'secondary_wn': 1601.0,
        'is_spurious_test': True,
        'spurious_type': 'weathering_carbonyl',
        'model_proxy': 'Carbonyl_peak'
    },
    'Spurious_Mineral_Carbonate_1420': {
        'family': 'Spurious Peak & False-Positive Tests',
        'title': 'Spurious Calcite/Dolomite Mineral Carbonate Intrusion (1420–1435 cm⁻¹)',
        'polymers': ['PE', 'PS', 'PET'],
        'confounders': ['PVC', 'PC'],
        'distinguishes': 'Tests FALSE-POSITIVE appearance & peak merging: Under-subtracted sediment carbonate (CaCO₃ at 1425 cm⁻¹) intrudes next to CH₂/Aromatic bands, mimicking PVC (1428 cm⁻¹) or distorting CH₂ scissoring.',
        'diagnostic_question': 'At what point along t does the intruding 1425 cm⁻¹ mineral carbonate band distort or dominate the native polymer peaks in the 1400–1500 cm⁻¹ window?',
        'visual_cues': [
            'Focus on the yellow corridor at 1410–1445 cm⁻¹.',
            'As t increases, a sharp/broad calcite carbonate band rises at 1425 cm⁻¹ with a companion 875/1080 tail.',
            'Mark the boundary t where the spurious 1425 cm⁻¹ mineral peak distorts or overtakes the polymer fingerprint.'
        ],
        'window': [1320.0, 1560.0],
        'target_ranges': [[1410.0, 1445.0]],
        'primary_wn': 1425.0,
        'secondary_wn': 1464.0,
        'is_spurious_test': True,
        'spurious_type': 'mineral_carbonate',
        'model_proxy': 'PE_doublet'
    }
}

FEATURE_NAMES = list(COMPREHENSIVE_FEATURES.keys())
NUM_FEATURES = len(FEATURE_NAMES)

# Map legacy 7 features for backwards compatibility
POLY_FEATURE_MAP = {
    'PE': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'PP': [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'PET': [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    'PLA': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'PVC': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Nylon': [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    'PS': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'ABS': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    'PC': [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    'PMMA': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'PU': [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
}


def compress_wavenumbers(wn):
    wn = np.asarray(wn)
    compressed = np.zeros_like(wn, dtype=float)
    seg1 = wn >= 2000.0
    compressed[seg1] = (4000.0 - wn[seg1]) / 1000.0
    seg2 = wn < 2000.0
    compressed[seg2] = (4000.0 - 2000.0) / 1000.0 + (2000.0 - wn[seg2]) / 200.0
    return compressed


def compute_second_derivative(spec: np.ndarray) -> np.ndarray:
    return savgol_filter(spec, window_length=15, polyorder=2, deriv=2)


def norm_spec(s: np.ndarray) -> np.ndarray:
    s_min = np.percentile(s, 2)
    s_clean = s - s_min
    s_max = np.percentile(s_clean, 99.5)
    return np.clip(s_clean / max(1e-4, s_max), 0.0, 1.2)


def compute_physical_perceptibility_scores(spec: np.ndarray, poly_class: str, noise_sigma: float) -> dict:
    """Computes physical SNR contrast scores for all diagnostic features."""
    noise_floor = max(1e-4, noise_sigma)
    scores = {}

    for feat_key, fmeta in COMPREHENSIVE_FEATURES.items():
        p_wn = fmeta.get('primary_wn', 1715.0)
        idx_p = np.argmin(np.abs(WN_GRID - p_wn))
        win = fmeta.get('window', [1200.0, 4000.0])
        m_win = (WN_GRID >= win[0]) & (WN_GRID <= win[1])
        local_base = np.percentile(spec[m_win], 15) if np.any(m_win) else 0.0

        if feat_key in ('PE_CH2_doublet', 'PE_doublet'):
            idx_2916 = np.argmin(np.abs(WN_GRID - 2916.0))
            idx_2848 = np.argmin(np.abs(WN_GRID - 2848.0))
            idx_trough = np.argmin(np.abs(WN_GRID - 2880.0))
            dip_depth = min(spec[idx_2916], spec[idx_2848]) - spec[idx_trough]
            c_ratio = dip_depth / (2.5 * noise_floor)
        else:
            prom = spec[idx_p] - local_base
            c_ratio = prom / (2.8 * noise_floor)

        scores[feat_key] = float(np.clip(1.0 / (1.0 + np.exp(-3.0 * (c_ratio - 1.0))), 0.0, 1.0))

    return scores


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_out = out_channels // 4
        self.b1 = nn.Conv1d(in_channels, branch_out, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_channels, branch_out, kernel_size=7, padding=3)
        self.b3 = nn.Conv1d(in_channels, branch_out, kernel_size=15, padding=7)
        self.b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_out, kernel_size=1)
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.relu(self.bn(out))


class FTIRPerceptibilityModel(nn.Module):
    def __init__(self, in_channels: int = 2, num_features: int = NUM_MODEL_FEATURES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.inc1 = InceptionBlock1D(32, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.inc2 = InceptionBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_features)
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.pool1(self.inc1(h))
        h = self.pool2(self.inc2(h))
        h = self.global_pool(h).squeeze(-1)
        logits = self.fc(h)
        return torch.sigmoid(logits)


# ------------------------------------------------------------------------------
# 1. Self-Contained Bundle Loader (Portable across GitHub clones & Local D: drive)
# ------------------------------------------------------------------------------
CONFIG_PATH = os.path.join(ROOT_DIR, "assets", "environmental_cohort_distortion_parameters.yaml")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(DATA_DIR, "environmental_cohort_distortion_parameters.yaml")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        ENV_CONFIG = yaml.safe_load(f)
    DISTRIBUTIONS = ENV_CONFIG.get('parameter_distributions', {})
else:
    ENV_CONFIG = {}
    DISTRIBUTIONS = {}


def _build_or_load_reference_bundle():
    """
    Loads polymer templates and natural/empirical backgrounds from D:\\user_data if available
    and caches a compact portable bundle in perceptibility_app/data/reference_bundle.npz
    so the app runs anywhere when cloned from GitHub.
    """
    clean_cache_path = r"D:\user_data\smolen\260923_MP_Parametric_Distortion_Profiling\clean_templates_and_real_particles.pkl"
    cat_path = r"D:\user_data\smolen\FTIR_Plastic_Datasets\Transformed_Similarity_CSVs\catalog.csv"
    by_id_dir = r"D:\user_data\smolen\FTIR_Plastic_Datasets\Transformed_Similarity_CSVs\by_id"
    non_mp_path = r"D:\user_data\smolen\261007_MP_clean_dataset_non-MP_detection\non_mp_clustered_spectra.npy"
    wn_882_path = r"D:\user_data\smolen\similarity_models\orig_wavenumbers.npy"
    nat_bkgd_dir = r"D:\user_data\smolen\260822_Bkgd_CSVs"

    if os.path.exists(BUNDLE_PATH):
        data = np.load(BUNDLE_PATH, allow_pickle=True)
        templates = {k: data[f"tmpl_{k}"] for k in ['PE', 'PP', 'PET', 'PLA', 'PVC', 'Nylon', 'PS', 'ABS', 'PC', 'PMMA', 'PU']}
        bg_pool = data["bg_pool"]
        bg_names = data["bg_names"].tolist()
        return templates, bg_pool, bg_names

    # Build from D:\user_data
    with open(clean_cache_path, "rb") as f:
        clean_cache = pickle.load(f)
    ref_templates = clean_cache['REFERENCE_TEMPLATES']
    cat_df = pd.read_csv(cat_path)

    def _load_cat(poly_class: str) -> np.ndarray:
        sub = cat_df[cat_df['polymer_class'] == poly_class]
        best_row = sub.sort_values(by=['interpretable_score', 'absorbance_max'], ascending=[False, False]).iloc[0]
        df = pd.read_csv(os.path.join(by_id_dir, best_row['file_name']))
        f_interp = interp1d(df['wavenumber'], df['absorbance'], kind='linear', fill_value='extrapolate')
        spec = f_interp(WN_GRID)
        spec = spec - np.percentile(spec[WN_GRID > 3800], 10)
        return np.maximum(0.0, spec)

    templates = {
        'PE': norm_spec(ref_templates['PE']),
        'PP': norm_spec(ref_templates['PP']),
        'PET': norm_spec(ref_templates['PET']),
        'PLA': norm_spec(ref_templates['PLA']),
        'PVC': norm_spec(ref_templates['PVC']),
        'Nylon': norm_spec(ref_templates['Nylon']),
        'PS': norm_spec(_load_cat('PS')),
        'ABS': norm_spec(_load_cat('ABS')),
        'PC': norm_spec(_load_cat('PC')),
        'PMMA': norm_spec(_load_cat('PMMA')),
        'PU': norm_spec(_load_cat('PU'))
    }

    wn_882 = np.load(wn_882_path)
    non_mp_882 = np.load(non_mp_path)
    f_bg = interp1d(wn_882, non_mp_882, axis=1, kind='linear', fill_value='extrapolate')
    non_mp_pool = f_bg(WN_GRID)

    nat_pool = []
    nat_names = []
    if os.path.exists(nat_bkgd_dir):
        import glob
        for cf in sorted(glob.glob(os.path.join(nat_bkgd_dir, "*.csv"))):
            try:
                bdf = pd.read_csv(cf)
                f_nat = interp1d(bdf['wavenumber'], bdf['absorbance'], kind='linear', fill_value='extrapolate')
                nat_pool.append(np.maximum(0.0, f_nat(WN_GRID)))
                nat_names.append(os.path.basename(cf))
            except Exception:
                pass

    # Save 16 natural backgrounds + 200 diverse non-MP backgrounds into portable bundle
    np.random.seed(42)
    sample_indices = np.linspace(0, len(non_mp_pool) - 1, 200, dtype=int)
    sampled_non_mp = non_mp_pool[sample_indices]
    non_mp_names = [f"riverbed_non_mp_{idx}.csv" for idx in sample_indices]

    if len(nat_pool) > 0:
        bg_pool = np.vstack([np.array(nat_pool), sampled_non_mp]).astype(np.float32)
        bg_names = nat_names + non_mp_names
    else:
        bg_pool = sampled_non_mp.astype(np.float32)
        bg_names = non_mp_names

    save_dict = {f"tmpl_{k}": v.astype(np.float32) for k, v in templates.items()}
    save_dict["bg_pool"] = bg_pool
    save_dict["bg_names"] = np.array(bg_names)
    np.savez_compressed(BUNDLE_PATH, **save_dict)
    print(f"[Engine] Created portable reference bundle at {BUNDLE_PATH}")
    return templates, bg_pool, bg_names


TEMPLATES, BG_POOL, BG_NAMES = _build_or_load_reference_bundle()

# Synthetic Biofilm, Hydration & Spurious Interference Profiles
WATER_PROFILE = np.exp(-((WN_GRID - 3350.0) / 200.0)**2) + 0.45 * np.exp(-((WN_GRID - 1640.0) / 55.0)**2)

BIOFILM_PROFILE = (
    0.70 * np.exp(-((WN_GRID - 3350.0) / 220.0)**2) +       # Hydration OH
    0.55 * np.exp(-((WN_GRID - 1648.0) / 42.0)**2) +        # Protein Amide I
    0.45 * np.exp(-((WN_GRID - 1545.0) / 38.0)**2) +        # Protein Amide II
    0.25 * np.exp(-((WN_GRID - 1405.0) / 50.0)**2) +        # Carboxylate sym
    0.55 * np.exp(-((WN_GRID - 1060.0) / 90.0)**2)          # Polysaccharide C-O-C broad
)

WEATHERING_CARBONYL_PROFILE = (
    0.85 * np.exp(-((WN_GRID - 1718.0) / 28.0)**2) +        # Ketone / Acid C=O
    0.35 * np.exp(-((WN_GRID - 1740.0) / 35.0)**2) +        # Ester / Lactone shoulder
    0.30 * np.exp(-((WN_GRID - 1175.0) / 65.0)**2) +        # C-O weathering hump
    0.25 * np.exp(-((WN_GRID - 3420.0) / 180.0)**2)         # Hydroperoxide / OH
)

CALCITE_CARBONATE_PROFILE = (
    1.00 * np.exp(-((WN_GRID - 1425.0) / 32.0)**2) +        # Asymmetric CO3 stretch
    0.15 * np.exp(-((WN_GRID - 2512.0) / 25.0)**2) +        # Carbonate combination band
    0.20 * np.exp(-((WN_GRID - 1795.0) / 20.0)**2)          # Carbonate overtone
)

# Load trained Perceptibility Model (supports local D: or portable data/ folder)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS_D = r"D:\user_data\smolen\261011_FTIR_Perceptibility_Models\ftir_perceptibility_model.pt"
MODEL_WEIGHTS_LOCAL = os.path.join(DATA_DIR, "ftir_perceptibility_model.pt")

if os.path.exists(MODEL_WEIGHTS_D) and not os.path.exists(MODEL_WEIGHTS_LOCAL):
    import shutil
    shutil.copy2(MODEL_WEIGHTS_D, MODEL_WEIGHTS_LOCAL)

WEIGHTS_TO_USE = MODEL_WEIGHTS_LOCAL if os.path.exists(MODEL_WEIGHTS_LOCAL) else MODEL_WEIGHTS_D
PERCEPTIBILITY_MODEL = FTIRPerceptibilityModel(in_channels=2, num_features=NUM_MODEL_FEATURES).to(DEVICE)
if os.path.exists(WEIGHTS_TO_USE):
    PERCEPTIBILITY_MODEL.load_state_dict(torch.load(WEIGHTS_TO_USE, map_location=DEVICE, weights_only=True))
    PERCEPTIBILITY_MODEL.eval()
    print("[Engine] Loaded trained FTIRPerceptibilityModel successfully.")
else:
    print("[Engine WARNING] Model weights not found, using uninitialized model.")


# Kramers-Kronig anomalous dispersion helper
def compute_kramers_kronig_dispersion(kappa: np.ndarray) -> np.ndarray:
    N = len(kappa)
    padded_N = 2048
    pad_spec = np.pad(kappa, (0, padded_N - N), mode='constant')
    F_spec = np.fft.fft(pad_spec)
    H = np.zeros(padded_N, dtype=np.complex128)
    H[1:padded_N // 2] = -1j
    H[padded_N // 2 + 1:] = 1j
    dispersion = np.real(np.fft.ifft(F_spec * H))[:N]
    return dispersion


def van_de_hulst_extinction(rho: np.ndarray) -> np.ndarray:
    rho = np.maximum(rho, 1e-4)
    return 2.0 - (4.0 / rho) * np.sin(rho) + (4.0 / (rho**2)) * (1.0 - np.cos(rho))


# ------------------------------------------------------------------------------
# 2. Physics-Based Forward Distortion & Spurious Feature Synthesizer
# ------------------------------------------------------------------------------
def synthesize_spectrum(
    polymer: str,
    t: float,
    mode: str = "compound",
    bg_idx: int = 42,
    seed: int = 1234,
    target_feature: str = "PE_CH2_doublet"
) -> dict:
    """
    Synthesizes degraded spectrum along parameterized trajectory t in [0.0, 1.0].
    t = 0.0: Pristine reference
    t = 1.0: Severe degradation / maximum spurious intrusion.
    """
    clean = TEMPLATES[polymer]
    np.random.seed(seed)

    fmeta = COMPREHENSIVE_FEATURES.get(target_feature, {})
    is_spurious_test = fmeta.get('is_spurious_test', False)
    spurious_type = fmeta.get('spurious_type', None)

    # Precompute anomalous dispersion curve for this polymer
    kappa = clean / (np.max(clean) + 1e-8)
    delta_n = compute_kramers_kronig_dispersion(kappa)
    r_eff = 12.0 * 1e-4  # 12 um effective radius
    n_rel = 1.45 + 0.3 * delta_n
    phase_rho = 4.0 * np.pi * r_eff * WN_GRID * np.abs(n_rel - 1.0)
    q_mie = van_de_hulst_extinction(phase_rho)
    q_mie_norm = (q_mie - q_mie.min()) / max(1e-4, q_mie.max() - q_mie.min())

    # Resolve parameters along trajectory t
    tau = 0.8
    rho_0 = 0.005
    snr = 80.0
    bsr = 0.0
    slope = 0.0
    clip_mismatch = 0.0
    sub_ratio = 1.0       # 1.0 = balanced, <1.0 = under-subtracted, >1.0 = over-subtracted
    sat_factor = 0.0      # 0.0 = linear, 1.0 = flat plateau detector saturation
    biofilm_amp = 0.0     # 0.0 = clean, >0.0 = water/EPS overlap
    spurious_amp = 0.0    # 0.0 = none, >0.0 = spurious diagnostic mimic peak amplitude

    if is_spurious_test or mode == "spurious_intrusion":
        # Trajectory specifically grows a false-positive diagnostic peak in a silent/confounding region
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.45]))
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.005, 0.12]))
        snr = float(np.interp(t, [0.0, 1.0], [110.0, 18.0]))
        bsr = float(np.interp(t, [0.0, 1.0], [0.2, 2.0]))
        spurious_amp = float(np.interp(t, [0.0, 1.0], [0.0, 0.85]))
    elif mode == "compound":
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.15]))
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.005, 0.45]))
        snr = float(np.interp(t, [0.0, 1.0], [120.0, 2.0]))
        bsr = float(np.interp(t, [0.0, 1.0], [0.5, 12.0]))
        slope = float(np.interp(t, [0.0, 1.0], [0.0, -0.35]))
        clip_mismatch = float(np.interp(t, [0.0, 1.0], [0.0, 0.25]))
        sub_ratio = float(np.interp(t, [0.0, 1.0], [1.0, 0.82]))
        biofilm_amp = float(np.interp(t, [0.0, 1.0], [0.0, 0.25]))
    elif mode == "pure_noise":
        tau = 0.8
        rho_0 = 0.005
        snr = float(np.interp(t, [0.0, 1.0], [150.0, 1.5]))
    elif mode == "pure_mie":
        tau = 0.8
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.002, 0.85]))
        snr = 80.0
        slope = float(np.interp(t, [0.0, 1.0], [0.0, -0.45]))
    elif mode == "pure_background":
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.2]))
        rho_0 = 0.005
        snr = 80.0
        bsr = float(np.interp(t, [0.0, 1.0], [0.1, 20.0]))
        clip_mismatch = float(np.interp(t, [0.0, 1.0], [0.0, 0.35]))
    elif mode == "under_subtracted":
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.35]))
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.01, 0.15]))
        snr = float(np.interp(t, [0.0, 1.0], [90.0, 8.0]))
        bsr = float(np.interp(t, [0.0, 1.0], [0.5, 6.0]))
        slope = float(np.interp(t, [0.0, 1.0], [0.0, -0.15]))
        sub_ratio = float(np.interp(t, [0.0, 1.0], [1.0, 0.25]))
    elif mode == "over_subtracted":
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.35]))
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.01, 0.15]))
        snr = float(np.interp(t, [0.0, 1.0], [90.0, 8.0]))
        bsr = float(np.interp(t, [0.0, 1.0], [0.5, 5.0]))
        slope = float(np.interp(t, [0.0, 1.0], [0.0, 0.15]))
        sub_ratio = float(np.interp(t, [0.0, 1.0], [1.0, 1.75]))
    elif mode == "peak_saturation":
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 4.5]))
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.01, 0.25]))
        snr = float(np.interp(t, [0.0, 1.0], [100.0, 15.0]))
        bsr = float(np.interp(t, [0.0, 1.0], [0.1, 1.5]))
        sat_factor = float(np.interp(t, [0.0, 1.0], [0.0, 0.95]))
    elif mode == "biofilm_hydration":
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.35]))
        rho_0 = float(np.interp(t, [0.0, 1.0], [0.01, 0.20]))
        snr = float(np.interp(t, [0.0, 1.0], [100.0, 10.0]))
        bsr = float(np.interp(t, [0.0, 1.0], [0.2, 2.5]))
        biofilm_amp = float(np.interp(t, [0.0, 1.0], [0.0, 1.4]))
    else:
        tau = float(np.interp(t, [0.0, 1.0], [0.8, 0.2]))

    # 1. Beer-Lambert Transmission & Optical Absorption
    s_abs = 1.0 - np.exp(-tau * kappa)

    # Apply Peak Saturation / Detector Dynamic Limit Clipping if sat_factor > 0
    if sat_factor > 0.01:
        t_stray = 0.002 + 0.12 * sat_factor
        a_nominal = s_abs * 2.2
        t_meas = 10.0**(-a_nominal) + t_stray
        s_abs = -np.log10(np.clip(t_meas, 1e-5, 1.0))
        s_abs = s_abs / max(1e-3, np.percentile(s_abs, 99.0)) * 0.85

    peak_abs = max(1e-3, np.max(s_abs))

    # 2. Resonant Mie Scattering Swing (with Kramers-Kronig derivative phase shift)
    s_pure = s_abs + rho_0 * peak_abs * (0.45 * q_mie_norm + 0.25 * delta_n)

    # 3. Biofilm & Hydration
    if biofilm_amp > 0.005:
        biofilm_spec = 0.55 * WATER_PROFILE + 0.45 * BIOFILM_PROFILE
        biofilm_spec = biofilm_spec / (np.max(biofilm_spec) + 1e-6)
        s_pure = s_pure + biofilm_amp * peak_abs * biofilm_spec

    # 4. Explicit Spurious Peak Intrusion (False-Positive Mimicry)
    if spurious_amp > 0.005:
        if spurious_type == 'biofilm_amide':
            sp_prof = BIOFILM_PROFILE
        elif spurious_type == 'weathering_carbonyl':
            sp_prof = WEATHERING_CARBONYL_PROFILE
        elif spurious_type == 'mineral_carbonate':
            sp_prof = CALCITE_CARBONATE_PROFILE
        else:
            # Default cycling spurious profile based on seed
            choice = seed % 3
            sp_prof = [BIOFILM_PROFILE, WEATHERING_CARBONYL_PROFILE, CALCITE_CARBONATE_PROFILE][choice]
        sp_prof = sp_prof / (np.max(sp_prof) + 1e-6)
        s_pure = s_pure + spurious_amp * peak_abs * sp_prof

    # 5. Baseline Slope
    wn_norm = (WN_GRID - 1196.0) / (4000.0 - 1196.0)
    baseline = slope * peak_abs * wn_norm

    # 6. Empirical Background Coupling
    bg_sample = BG_POOL[bg_idx % len(BG_POOL)]
    bg_clean = bg_sample - np.percentile(bg_sample, 5)
    bg_norm = bg_clean / (np.max(bg_clean) + 1e-6)
    bg_particle = bg_norm * (peak_abs * bsr)

    s_raw = s_pure + bg_particle + baseline

    # 7. Active-Mask Pink (1/f) Noise
    target_sigma = peak_abs / max(1.0, snr)
    pink_filter = 1.0 / np.sqrt(np.linspace(1, 10, len(WN_GRID)))
    raw_noise = np.random.normal(0, target_sigma, len(WN_GRID)) * pink_filter
    s_raw += raw_noise

    # 8. Annular Measurement with Subtraction Mismatch
    mismatch_factor = sub_ratio * (1.0 + 0.05 * clip_mismatch)
    bg_noise = np.random.normal(0, target_sigma * 0.3, len(WN_GRID))
    bg_drift = np.random.normal(0, 0.01 * peak_abs)
    bg_annulus = bg_particle * mismatch_factor + bg_noise + bg_drift

    # 9. Annular Subtraction
    s_sub_raw = s_raw - bg_annulus
    s_sub = np.maximum(-0.25 * peak_abs, s_sub_raw)

    s_norm = norm_spec(s_sub)
    d2 = compute_second_derivative(s_sub)
    d2_norm = d2 / (np.max(np.abs(d2)) + 1e-5)

    # 10. Real-time Model Inference
    x_ten = torch.tensor(np.stack([s_norm, d2_norm])[np.newaxis, ...], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        raw_probs = PERCEPTIBILITY_MODEL(x_ten)[0].cpu().numpy()

    legacy_probs = {MODEL_FEATURE_NAMES[i]: float(raw_probs[i]) for i in range(NUM_MODEL_FEATURES)}
    model_predictions = {}
    for fk, fm in COMPREHENSIVE_FEATURES.items():
        proxy = fm.get('model_proxy', 'PE_doublet')
        model_predictions[fk] = legacy_probs.get(proxy, 0.5)
    # Also include legacy keys
    model_predictions.update(legacy_probs)

    # Physical scores
    gt_scores = compute_physical_perceptibility_scores(s_sub, polymer, target_sigma)

    zeta_zero = float(np.mean(s_sub <= 1e-4))
    comp_wns = compress_wavenumbers(WN_GRID)
    bg_name = BG_NAMES[bg_idx % len(BG_NAMES)]

    return {
        'polymer': polymer,
        't': float(t),
        'mode': mode,
        'bg_name': bg_name,
        'is_spurious_test': is_spurious_test,
        'wavenumbers': WN_GRID.tolist(),
        'compressed_wavenumbers': comp_wns.tolist(),
        'absorbance': s_norm.tolist(),
        'second_derivative': d2_norm.tolist(),
        'raw_absorbance': s_raw.tolist(),
        'background_absorbance': bg_annulus.tolist(),
        'params': {
            'tau': float(tau),
            'rho_0': float(rho_0),
            'snr': float(snr),
            'bsr': float(bsr),
            'slope': float(slope),
            'zeta_zero': float(zeta_zero),
            'sub_ratio': float(sub_ratio),
            'sat_factor': float(sat_factor),
            'biofilm_amp': float(biofilm_amp),
            'spurious_amp': float(spurious_amp),
            'sigma_noise': float(target_sigma)
        },
        'model_predictions': model_predictions,
        'physical_scores': gt_scores
    }
