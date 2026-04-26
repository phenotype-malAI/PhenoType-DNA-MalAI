"""
PHENO TYPE — attribute.py
Attribution engine: loads trained model + centroids, classifies a 1200-token sequence.

Usage:
    python attribute.py --csv_row final_dna_v2.csv --row_idx 0
    python attribute.py --tokens "3 22 3 2 ..."
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import BehaviourEncoder
from dataset import IDX_TO_FAMILY, FAMILY_TO_IDX

# Update this after running train.py — check threshold_calibration.png
# First run suggested 0.99, but use 0.97 to avoid over-rejection
THRESHOLD = 0.986


def load_model_and_centroids(encoder_path='outputs/behaviour_encoder.pt',
                              centroids_path='outputs/family_centroids.pt',
                              device='cpu'):
    device = torch.device(device)
    model  = BehaviourEncoder()
    model.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=False))
    model.to(device).eval()
    centroids = torch.load(centroids_path, map_location='cpu', weights_only=False)
    return model, centroids, device


def attribute(token_sequence, model, centroids, device, threshold=THRESHOLD,
              return_attn=False):
    model.eval()
    if token_sequence.dim() == 1:
        token_sequence = token_sequence.unsqueeze(0)
    token_sequence = token_sequence.to(device)

    with torch.no_grad():
        if return_attn:
            fp, attn = model(token_sequence, return_attn_weights=True)
            attn = attn.squeeze(0).cpu()
        else:
            fp    = model(token_sequence)
            attn  = None

    fp = fp.squeeze(0)

    all_scores = {}
    for fam_idx, centroid in centroids.items():
        sim = torch.dot(fp, centroid.to(device)).item()
        all_scores[IDX_TO_FAMILY[fam_idx]] = round(sim, 6)

    best_family = max(all_scores, key=all_scores.get)
    best_score  = all_scores[best_family]

    if best_score >= threshold:
        return best_family, best_score, all_scores, attn
    else:
        return 'UNKNOWN', best_score, all_scores, attn


def format_result(family, best_score, all_scores, threshold=THRESHOLD):
    sep = '─' * 48
    lines = [
        sep,
        f"  Predicted Family : {family}",
        f"  Confidence Score : {best_score:.4f}  "
        f"({'≥' if best_score >= threshold else '<'} threshold {threshold})",
        sep,
        "  All family similarities:",
    ]
    for fname, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        bar = '█' * int(score * 30)
        lines.append(f"    {fname:<14} {score:.4f}  {bar}")
    lines.append(sep)
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens',    type=str,  default=None)
    parser.add_argument('--csv_row',   type=str,  default=None)
    parser.add_argument('--row_idx',   type=int,  default=0)
    parser.add_argument('--encoder',   default='outputs/behaviour_encoder.pt')
    parser.add_argument('--centroids', default='outputs/family_centroids.pt')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--device',    default='cpu')
    parser.add_argument('--json_out',  action='store_true')
    args = parser.parse_args()

    model, centroids, device = load_model_and_centroids(
        args.encoder, args.centroids, args.device
    )

    if args.tokens:
        vals = list(map(int, args.tokens.strip().split()))
        if len(vals) != 1200:
            print(f"ERROR: expected 1200 tokens, got {len(vals)}", file=sys.stderr)
            sys.exit(1)
        tok = torch.tensor(vals, dtype=torch.long)

    elif args.csv_row:
        import pandas as pd
        df  = pd.read_csv(args.csv_row, nrows=args.row_idx + 1)
        row = df.iloc[args.row_idx]
        tok_cols = [f'tok_{i}' for i in range(1200)]
        tok = torch.tensor(row[tok_cols].values.astype('int64'), dtype=torch.long)
        print(f"Row {args.row_idx}: family={row['family']}  "
              f"sha256={str(row['sha256'])[:16]}…")
    else:
        parser.print_help()
        sys.exit(1)

    family, best_score, all_scores, _ = attribute(
        tok, model, centroids, device, threshold=args.threshold
    )

    if args.json_out:
        print(json.dumps({'family': family, 'score': best_score,
                          'all_scores': all_scores}, indent=2))
    else:
        print(format_result(family, best_score, all_scores, args.threshold))