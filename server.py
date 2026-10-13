"""
perceptibility_app/server.py
============================
HTTP JSON API and Static File Server for the Microplastics FTIR Perceptibility Study App.
Runs locally using Python's built-in standard library http.server on port 8055.
Supports Polymer Family -> Diagnostic Feature hierarchy, Spurious Peak tests,
Yellow Target Highlighting, and Dual-Storage Sync (local repo + D:\\user_data).
"""

import os
import sys
import json
import random
import shutil
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Set OpenMP environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure path to autoLUMOS
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from perceptibility_app.engine import (
    synthesize_spectrum,
    COMPREHENSIVE_FEATURES,
    FEATURE_NAMES,
    POLY_FEATURE_MAP,
    TEMPLATES,
    WN_GRID
)

STATIC_DIR = os.path.join(APP_DIR, "static")
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Dual-sync trial storage:
# 1. Portable repository file (always present for GitHub sharing & remote access)
REPO_TRIALS_PATH = os.path.join(DATA_DIR, "perceptibility_trials.jsonl")
# 2. Local D: drive master archive (synced when running on primary workstation)
EXTERNAL_TRIALS_PATH = r"D:\user_data\smolen\261011_FTIR_Perceptibility_Models\perceptibility_trials.jsonl"

# Initialize / synchronize on startup
if os.path.exists(EXTERNAL_TRIALS_PATH) and not os.path.exists(REPO_TRIALS_PATH):
    shutil.copy2(EXTERNAL_TRIALS_PATH, REPO_TRIALS_PATH)
elif os.path.exists(EXTERNAL_TRIALS_PATH) and os.path.exists(REPO_TRIALS_PATH):
    # Ensure repo file has latest external records if external is larger
    if os.path.getsize(EXTERNAL_TRIALS_PATH) > os.path.getsize(REPO_TRIALS_PATH):
        shutil.copy2(EXTERNAL_TRIALS_PATH, REPO_TRIALS_PATH)

# Copy environmental YAML to data/ for standalone portability
YAML_SRC = os.path.join(ROOT_DIR, "assets", "environmental_cohort_distortion_parameters.yaml")
YAML_DST = os.path.join(DATA_DIR, "environmental_cohort_distortion_parameters.yaml")
if os.path.exists(YAML_SRC) and not os.path.exists(YAML_DST):
    shutil.copy2(YAML_SRC, YAML_DST)

# Comprehensive Polymer Reference Peaks Dictionary (for non-target reference context)
POLYMER_IMPORTANT_PEAKS = {
    'PE': [
        {'wn': 2916.0, 'range': [2900.0, 2932.0], 'label': '2916 (asym CH₂)', 'tier': 'Primary', 'role': 'Splitting doublet: PE vs PP'},
        {'wn': 2848.0, 'range': [2835.0, 2862.0], 'label': '2848 (sym CH₂)', 'tier': 'Primary', 'role': 'Splitting doublet: PE vs PP'},
        {'wn': 1464.0, 'range': [1450.0, 1480.0], 'label': '1464 (CH₂ scissoring)', 'tier': 'Secondary', 'role': 'Crystalline splitting (1472/1462)'},
        {'wn': 1377.0, 'range': [1365.0, 1390.0], 'label': '1377 (CH₃ umbrella)', 'tier': 'Negative / Discriminator', 'role': 'ABSENT in pure HDPE; tiny in LDPE'}
    ],
    'PP': [
        {'wn': 2950.0, 'range': [2938.0, 2968.0], 'label': '2950 (asym CH₃)', 'tier': 'Primary', 'role': 'High-prominence methyl CH₃ vs PE'},
        {'wn': 2918.0, 'range': [2905.0, 2930.0], 'label': '2918 (asym CH₂)', 'tier': 'Primary', 'role': 'Envelope shoulder in cluster'},
        {'wn': 2873.0, 'range': [2860.0, 2885.0], 'label': '2873 (sym CH₃)', 'tier': 'Secondary', 'role': 'Diagnostic methyl shoulder'},
        {'wn': 2838.0, 'range': [2825.0, 2850.0], 'label': '2838 (sym CH₂)', 'tier': 'Secondary', 'role': 'Lower envelope cluster base'},
        {'wn': 1456.0, 'range': [1440.0, 1470.0], 'label': '1456 (CH₃ asym bend)', 'tier': 'Secondary', 'role': 'Intense backbone bending'},
        {'wn': 1377.0, 'range': [1362.0, 1392.0], 'label': '1377 (CH₃ umbrella)', 'tier': 'Primary', 'role': 'Strong symmetric umbrella (confirms PP vs PE)'}
    ],
    'PS': [
        {'wn': 3026.0, 'range': [3010.0, 3075.0], 'label': '3026 (Aromatic =C–H)', 'tier': 'Primary', 'role': 'Unsaturated aromatic ring stretch (>3000 cm⁻¹)'},
        {'wn': 1601.0, 'range': [1585.0, 1615.0], 'label': '1601 (Ring quadrant)', 'tier': 'Primary', 'role': 'Diagnostic mono-substituted phenyl ring'},
        {'wn': 1493.0, 'range': [1478.0, 1508.0], 'label': '1493 (Ring semicircle)', 'tier': 'Primary', 'role': 'Sharp aromatic ring vibration'},
        {'wn': 1452.0, 'range': [1435.0, 1465.0], 'label': '1452 (Ring + CH₂)', 'tier': 'Secondary', 'role': 'Combined aliphatic/aromatic mode'},
        {'wn': 2237.0, 'range': [2218.0, 2255.0], 'label': '2237 (Nitrile C≡N)', 'tier': 'Negative / Discriminator', 'role': 'ABSENT in pure PS; differentiates PS from ABS'}
    ],
    'ABS': [
        {'wn': 2237.0, 'range': [2218.0, 2255.0], 'label': '2237 (Nitrile C≡N)', 'tier': 'Primary', 'role': 'Crucial discriminator: ABS vs PS / SAN'},
        {'wn': 3026.0, 'range': [3010.0, 3075.0], 'label': '3026 (Aromatic =C–H)', 'tier': 'Secondary', 'role': 'Styrenic component stretch'},
        {'wn': 1601.0, 'range': [1585.0, 1615.0], 'label': '1601 (Ring quadrant)', 'tier': 'Secondary', 'role': 'Styrene quadrant stretch'},
        {'wn': 1493.0, 'range': [1478.0, 1508.0], 'label': '1493 (Ring semicircle)', 'tier': 'Secondary', 'role': 'Sharp aromatic ring marker'},
        {'wn': 1720.0, 'range': [1700.0, 1745.0], 'label': '1720 (Carbonyl C=O)', 'tier': 'Negative / Weathering', 'role': 'ABSENT in pure ABS; indicates photo-oxidation'}
    ],
    'PET': [
        {'wn': 1715.0, 'range': [1695.0, 1732.0], 'label': '1715 (Ester C=O)', 'tier': 'Primary', 'role': 'Intense ester carbonyl (conjugated with terephthalate)'},
        {'wn': 1245.0, 'range': [1210.0, 1265.0], 'label': '1245 (Ester C–O–C)', 'tier': 'Primary', 'role': 'Strong ester asymmetric stretch'},
        {'wn': 1410.0, 'range': [1395.0, 1425.0], 'label': '1410 (Aromatic ring)', 'tier': 'Secondary', 'role': 'para-substituted terephthalate ring'},
        {'wn': 1505.0, 'range': [1490.0, 1520.0], 'label': '1505 (Aromatic C=C)', 'tier': 'Secondary', 'role': 'Diagnostic terephthalate ring vibration'}
    ],
    'PLA': [
        {'wn': 1750.0, 'range': [1732.0, 1768.0], 'label': '1750 (Aliphatic C=O)', 'tier': 'Primary', 'role': 'High-frequency aliphatic ester (PLA vs PET 1715)'},
        {'wn': 1452.0, 'range': [1435.0, 1470.0], 'label': '1452 (CH₃ asym)', 'tier': 'Secondary', 'role': 'Methyl deformation in lactide backbone'},
        {'wn': 1360.0, 'range': [1340.0, 1380.0], 'label': '1360 (CH bend)', 'tier': 'Secondary', 'role': 'Methine C–H bending'},
        {'wn': 1210.0, 'range': [1196.0, 1235.0], 'label': '1210 (C–O ester)', 'tier': 'Secondary', 'role': 'Backbone aliphatic ester stretch'}
    ],
    'Nylon': [
        {'wn': 3300.0, 'range': [3265.0, 3345.0], 'label': '3300 (Amide A / N–H)', 'tier': 'Primary', 'role': 'H-bonded secondary amide stretch'},
        {'wn': 1640.0, 'range': [1620.0, 1660.0], 'label': '1640 (Amide I / C=O)', 'tier': 'Primary', 'role': 'Strong amide carbonyl stretching'},
        {'wn': 1540.0, 'range': [1520.0, 1562.0], 'label': '1540 (Amide II / N–H+C–N)', 'tier': 'Primary', 'role': 'Diagnostic secondary amide II (confirms polyamide)'},
        {'wn': 2928.0, 'range': [2905.0, 2950.0], 'label': '2928 (asym CH₂)', 'tier': 'Secondary', 'role': 'Aliphatic polymethylene segments'}
    ],
    'PU': [
        {'wn': 3320.0, 'range': [3265.0, 3360.0], 'label': '3320 (Urethane N–H)', 'tier': 'Primary', 'role': 'H-bonded urethane amine stretch'},
        {'wn': 1725.0, 'range': [1700.0, 1748.0], 'label': '1725 (Urethane C=O)', 'tier': 'Primary', 'role': 'Urethane carbonyl (ester/ether linkage)'},
        {'wn': 1530.0, 'range': [1510.0, 1555.0], 'label': '1530 (Amide/Urethane II)', 'tier': 'Primary', 'role': 'N–H bend / C–N stretch coupling'},
        {'wn': 1225.0, 'range': [1205.0, 1250.0], 'label': '1225 (C–O–C stretch)', 'tier': 'Secondary', 'role': 'Polyether or polyester soft segment'}
    ],
    'PC': [
        {'wn': 1775.0, 'range': [1755.0, 1795.0], 'label': '1775 (Carbonate C=O)', 'tier': 'Primary', 'role': 'High-frequency carbonate carbonyl (PC vs PET/PLA)'},
        {'wn': 1504.0, 'range': [1490.0, 1518.0], 'label': '1504 (Bisphenol A ring)', 'tier': 'Primary', 'role': 'Sharp bisphenol-A aromatic skeletal band'},
        {'wn': 1225.0, 'range': [1205.0, 1255.0], 'label': '1225 (Carbonate C–O–C)', 'tier': 'Secondary', 'role': 'Carbonate ester linkage'}
    ],
    'PMMA': [
        {'wn': 1730.0, 'range': [1712.0, 1748.0], 'label': '1730 (Methacrylate C=O)', 'tier': 'Primary', 'role': 'Strong ester carbonyl of methyl methacrylate'},
        {'wn': 1450.0, 'range': [1430.0, 1470.0], 'label': '1450 (CH₃ bending)', 'tier': 'Secondary', 'role': 'Ester methyl & α-methyl bending'},
        {'wn': 1240.0, 'range': [1215.0, 1265.0], 'label': '1240 (C–O–C ester)', 'tier': 'Secondary', 'role': 'Ester oxygen skeletal stretching'}
    ],
    'PVC': [
        {'wn': 2920.0, 'range': [2890.0, 2950.0], 'label': '2920 (CH₂ / CH–Cl)', 'tier': 'Secondary', 'role': 'Chlorinated backbone aliphatic stretch'},
        {'wn': 1428.0, 'range': [1412.0, 1442.0], 'label': '1428 (CH₂–Cl bend)', 'tier': 'Primary', 'role': 'Diagnostic methylene next to chlorine'},
        {'wn': 1255.0, 'range': [1238.0, 1272.0], 'label': '1255 (CH–Cl rocking)', 'tier': 'Primary', 'role': 'In-plane rocking of CH–Cl'}
    ]
}

DEGRADATION_MODES = [
    'compound',           # Realistic multi-component environmental physics
    'under_subtracted',   # Natural substrate mineral/silt positive bleed-through
    'over_subtracted',    # Over-subtracted annular background (negative troughs)
    'peak_saturation',    # Optical thickness saturation & plateau clipping
    'biofilm_hydration',  # Water OH/bend & microbial EPS broad overlap
    'spurious_intrusion', # False-positive diagnostic band intrusion
    'pure_noise',         # Pink 1/f instrument noise degradation only
    'pure_mie',           # Resonant anomalous dispersion Mie swing only
    'pure_background'    # Empirical annular sediment/mineral coupling only
]


def _sync_external_trials():
    """Keep external D: drive file synchronized if accessible."""
    try:
        ext_dir = os.path.dirname(EXTERNAL_TRIALS_PATH)
        if os.path.exists(ext_dir) and os.path.exists(REPO_TRIALS_PATH):
            shutil.copy2(REPO_TRIALS_PATH, EXTERNAL_TRIALS_PATH)
    except Exception:
        pass


class PerceptibilityRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/trial":
            self.handle_api_trial(parsed)
        elif path == "/api/taxonomy":
            self.handle_api_taxonomy()
        elif path == "/api/polymers":
            self.handle_api_polymers()
        elif path == "/api/features":
            self.handle_api_features()
        elif path == "/api/history":
            self.handle_api_history()
        elif path.startswith("/api/"):
            self.send_error(404, "API endpoint not found")
        else:
            self.serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            body = json.loads(post_data.decode('utf-8'))
        except Exception:
            body = {}

        if path == "/api/synthesize":
            self.handle_api_synthesize(body)
        elif path == "/api/record":
            self.handle_api_record(body)
        elif path == "/api/delete_trial":
            self.handle_api_delete_trial(body)
        else:
            self.send_error(404, "API endpoint not found")

    def handle_api_taxonomy(self):
        """Returns the full Polymer Family -> Diagnostic Feature hierarchy."""
        families = {}
        for feat_key, fmeta in COMPREHENSIVE_FEATURES.items():
            fam = fmeta['family']
            if fam not in families:
                families[fam] = []
            families[fam].append({
                'key': feat_key,
                'title': fmeta['title'],
                'polymers': fmeta['polymers'],
                'confounders': fmeta.get('confounders', []),
                'distinguishes': fmeta['distinguishes'],
                'diagnostic_question': fmeta['diagnostic_question'],
                'visual_cues': fmeta['visual_cues'],
                'window': fmeta['window'],
                'target_ranges': fmeta.get('target_ranges', []),
                'is_spurious_test': fmeta.get('is_spurious_test', False)
            })
        self.send_json({
            'families': families,
            'features': COMPREHENSIVE_FEATURES
        })

    def handle_api_polymers(self):
        polys = list(TEMPLATES.keys())
        self.send_json({
            'polymers': polys,
            'feature_map': POLY_FEATURE_MAP,
            'polymer_peaks': POLYMER_IMPORTANT_PEAKS
        })

    def handle_api_features(self):
        feats = []
        for feat_key, info in COMPREHENSIVE_FEATURES.items():
            feats.append({
                'name': feat_key,
                'family': info.get('family', ''),
                'title': info.get('title', feat_key),
                'polymers': info.get('polymers', []),
                'distinguishes': info.get('distinguishes', ''),
                'window': info.get('window', [1200.0, 4000.0]),
                'target_ranges': info.get('target_ranges', []),
                'diagnostic_question': info.get('diagnostic_question', ''),
                'visual_cues': info.get('visual_cues', []),
                'is_spurious_test': info.get('is_spurious_test', False)
            })
        self.send_json({'features': feats})

    def handle_api_trial(self, parsed):
        qs = parse_qs(parsed.query)
        req_feature = qs.get('feature', [None])[0]
        req_family = qs.get('family', [None])[0]
        req_mode = qs.get('mode', [None])[0]

        # 1. Select Feature (either requested, or sampled from requested family, or random)
        if req_feature and req_feature in COMPREHENSIVE_FEATURES:
            target_feature = req_feature
        elif req_family:
            fam_feats = [k for k, v in COMPREHENSIVE_FEATURES.items() if v['family'] == req_family]
            target_feature = random.choice(fam_feats) if fam_feats else random.choice(FEATURE_NAMES)
        else:
            target_feature = random.choice(FEATURE_NAMES)

        feat_info = COMPREHENSIVE_FEATURES[target_feature]

        # 2. Automatically draw a relevant polymer for this feature!
        candidate_polys = feat_info.get('polymers', ['PE'])
        polymer = random.choice(candidate_polys)

        # 3. Determine degradation mode
        if feat_info.get('is_spurious_test', False):
            mode = 'spurious_intrusion'
        elif req_mode and req_mode in DEGRADATION_MODES:
            mode = req_mode
        else:
            # Draw from realistic physics modes
            standard_modes = ['compound', 'under_subtracted', 'over_subtracted', 'peak_saturation', 'biofilm_hydration', 'pure_noise', 'pure_mie']
            mode = random.choice(standard_modes)

        seed = random.randint(100, 999999)
        bg_idx = random.randint(0, 999)

        important_peaks = POLYMER_IMPORTANT_PEAKS.get(polymer, [])
        clean_template = TEMPLATES[polymer]
        clean_abs = clean_template / (float(clean_template.max()) + 1e-6)

        trial_data = {
            'target_feature': target_feature,
            'family': feat_info.get('family', ''),
            'feature_title': feat_info.get('title', target_feature),
            'distinguishes': feat_info.get('distinguishes', ''),
            'confounders': feat_info.get('confounders', []),
            'candidate_polymers': candidate_polys,
            'diagnostic_question': feat_info.get('diagnostic_question', ''),
            'visual_cues': feat_info.get('visual_cues', []),
            'primary_wavenumber': feat_info.get('primary_wn', None),
            'secondary_wavenumber': feat_info.get('secondary_wn', None),
            'feature_window': feat_info.get('window', [1200.0, 4000.0]),
            'target_ranges': feat_info.get('target_ranges', []),
            'is_spurious_test': feat_info.get('is_spurious_test', False),
            'polymer': polymer,
            'important_peaks': important_peaks,
            'clean_reference_absorbance': clean_abs.tolist(),
            'mode': mode,
            'seed': seed,
            'bg_idx': bg_idx,
            't_default': 0.0
        }
        self.send_json(trial_data)

    def handle_api_synthesize(self, body):
        polymer = body.get('polymer', 'PE')
        t = float(body.get('t', 0.0))
        mode = body.get('mode', 'compound')
        bg_idx = int(body.get('bg_idx', 42))
        seed = int(body.get('seed', 1234))
        target_feature = body.get('target_feature', 'PE_CH2_doublet')

        try:
            result = synthesize_spectrum(
                polymer=polymer,
                t=t,
                mode=mode,
                bg_idx=bg_idx,
                seed=seed,
                target_feature=target_feature
            )
            result['important_peaks'] = POLYMER_IMPORTANT_PEAKS.get(polymer, [])
            self.send_json(result)
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

    def handle_api_record(self, body):
        does_not_degrade = bool(body.get('does_not_degrade', False))
        t_boundary_val = None if does_not_degrade else float(body.get('t_boundary', 0.5))
        is_spurious = bool(body.get('is_spurious_test', False))

        if is_spurious:
            outcome = 'no_spurious_appearance' if does_not_degrade else 'spurious_peak_perceived'
        else:
            outcome = 'observable_throughout' if does_not_degrade else 'perceptual_breakdown'

        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': body.get('user', 'smolen'),
            'polymer': body.get('polymer'),
            'family': body.get('family', ''),
            'target_feature': body.get('target_feature'),
            'is_spurious_test': is_spurious,
            'mode': body.get('mode'),
            'does_not_degrade': does_not_degrade,
            'outcome': outcome,
            't_boundary': t_boundary_val,
            'user_confidence': body.get('user_confidence', 'high'),
            'notes': body.get('notes', ''),
            'params_at_boundary': body.get('params', {}),
            'model_prediction_at_boundary': body.get('model_prediction', None),
            'physical_score_at_boundary': body.get('physical_score', None),
            'seed': body.get('seed'),
            'bg_idx': body.get('bg_idx')
        }

        try:
            with open(REPO_TRIALS_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
            _sync_external_trials()

            total_records = 0
            if os.path.exists(REPO_TRIALS_PATH):
                with open(REPO_TRIALS_PATH, 'r', encoding='utf-8') as f:
                    total_records = sum(1 for line in f if line.strip())

            self.send_json({'status': 'success', 'recorded_count': total_records, 'record': record})
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

    def handle_api_history(self):
        records = []
        if os.path.exists(REPO_TRIALS_PATH):
            try:
                with open(REPO_TRIALS_PATH, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            item['index'] = idx
                            records.append(item)
            except Exception as e:
                print("Error reading history:", e)

        tallies = {}
        outcomes = {}
        for r in records:
            feat = r.get('target_feature', 'unknown')
            tallies[feat] = tallies.get(feat, 0) + 1
            out = r.get('outcome', 'perceptual_breakdown')
            outcomes[out] = outcomes.get(out, 0) + 1

        self.send_json({
            'total_count': len(records),
            'tallies': tallies,
            'outcomes': outcomes,
            'records': records[-80:]
        })

    def handle_api_delete_trial(self, body):
        del_index = body.get('index', None)
        if del_index is None:
            self.send_error(400, "Missing trial index to delete")
            return

        del_index = int(del_index)
        all_lines = []
        if os.path.exists(REPO_TRIALS_PATH):
            with open(REPO_TRIALS_PATH, 'r', encoding='utf-8') as f:
                all_lines = [l for l in f if l.strip()]

        if 0 <= del_index < len(all_lines):
            deleted_line = all_lines.pop(del_index)
            with open(REPO_TRIALS_PATH, 'w', encoding='utf-8') as f:
                for l in all_lines:
                    f.write(l.strip() + "\n")
            _sync_external_trials()
            print(f"[Server] Deleted trial at index {del_index}: {deleted_line[:80]}...")
            self.send_json({'status': 'success', 'deleted_index': del_index, 'remaining_count': len(all_lines)})
        else:
            self.send_error(404, f"Index {del_index} out of range (total {len(all_lines)})")

    def serve_static(self, path):
        if path == "/" or path == "":
            path = "/index.html"

        file_path = os.path.join(STATIC_DIR, path.lstrip("/"))
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_error(404, f"File not found: {path}")
            return

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


def run_server(port=8055):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, PerceptibilityRequestHandler)
    print(f"Perceptibility Server running at http://127.0.0.1:{port}/")
    print(f"Logging trials to: {REPO_TRIALS_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Perceptibility Server.")
        httpd.server_close()


if __name__ == "__main__":
    port = 8055
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
