"""
plot_roc.py  —  Signal efficiency vs background rejection for the trained NN.

Reads test_results.npz from a run directory and plots:
  1. ROC curve: Lambda_c+ signal efficiency vs background rejection (1 - FPR)
  2. Optionally overlays ROC curves from multiple run directories for comparison.

Usage:
    python plot_roc.py <run_dir> [<run_dir2> ...]
    python plot_roc.py 20260325-113250 final5_10 --label "Full pT" "5-10 GeV"
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 1.2,
})

# Class labels (as written by event_generator + evaluate_model)
CLASS_SIG   = 2   # Lambda_c+
CLASS_CHARM = 1   # other charm (background)
CLASS_BG    = 0   # non-charm background


def load_results(run_dir, results_filename='test_results.npz'):
    path = os.path.join('runs', run_dir, results_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results not found: {path}")
    data = np.load(path)
    return data['y_true'].astype(int), data['y_proba']


def compute_roc(y_true, y_proba):
    """
    Binary ROC: Lambda_c+ (class 2) vs everything else.
    Returns fpr, tpr, auc_score.
    Signal score = y_proba[:, 2].
    """
    binary_true = (y_true == CLASS_SIG).astype(int)
    scores      = y_proba[:, CLASS_SIG]
    fpr, tpr, _ = roc_curve(binary_true, scores)
    auc_score   = auc(fpr, tpr)
    return fpr, tpr, auc_score


def main():
    parser = argparse.ArgumentParser(
        description="Plot signal efficiency vs background rejection (ROC curve)."
    )
    parser.add_argument('run_dirs', nargs='+', type=str,
                        help='One or more run directory names under runs/')
    parser.add_argument('--labels', nargs='+', type=str, default=None,
                        help='Custom labels for each run (must match number of run_dirs)')
    parser.add_argument('--results', type=str, default='test_results.npz',
                        help='NPZ filename inside each run directory')
    parser.add_argument('--output', '-o', default='roc_curve.png',
                        help='Output plot filename')
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.run_dirs):
        print("Error: --labels must have the same number of entries as run_dirs.")
        sys.exit(1)

    labels = args.labels if args.labels else args.run_dirs

    # Colour cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax_main, ax_eff) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    for i, (run_dir, label) in enumerate(zip(args.run_dirs, labels)):
        color = colors[i % len(colors)]
        try:
            y_true, y_proba = load_results(run_dir, args.results)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping.")
            continue

        fpr, tpr, auc_score = compute_roc(y_true, y_proba)

        # Avoid division by zero for FPR=0 points
        valid = fpr > 0
        bg_rejection = np.where(valid, 1.0 / fpr, np.nan)

        ax_main.plot(tpr, bg_rejection, color=color, linewidth=2.0,
                     label=rf'{label}  (AUC = {auc_score:.4f})')

        # Lower panel: background efficiency vs signal efficiency
        ax_eff.plot(tpr, fpr, color=color, linewidth=1.5, linestyle='-')

    # --- Top panel formatting ---
    ax_main.set_xlabel(r'Signal Efficiency  $\epsilon_S$', fontsize=15)
    ax_main.set_ylabel(r'Background Rejection  $1\,/\,\epsilon_B$', fontsize=15)
    ax_main.set_title(r'$\Lambda_c^+$ Signal Efficiency vs Background Rejection', fontsize=15)
    ax_main.set_xlim(0, 1)
    ax_main.set_yscale('log')
    ax_main.minorticks_on()
    ax_main.grid(True, which='major', linestyle='--', alpha=0.3, zorder=0)
    ax_main.grid(True, which='minor', linestyle=':',  alpha=0.2, zorder=0)
    ax_main.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8)
    ax_main.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4)
    ax_main.legend(loc='upper right', frameon=False)

    # --- Bottom panel: background efficiency (FPR) vs signal efficiency ---
    ax_eff.set_xlabel(r'Signal Efficiency  $\epsilon_S$', fontsize=15)
    ax_eff.set_ylabel(r'$\epsilon_B$', fontsize=14)
    ax_eff.set_xlim(0, 1)
    ax_eff.set_ylim(bottom=0)
    ax_eff.minorticks_on()
    ax_eff.grid(True, which='major', linestyle='--', alpha=0.3, zorder=0)
    ax_eff.grid(True, which='minor', linestyle=':',  alpha=0.2, zorder=0)
    ax_eff.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8)
    ax_eff.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4)

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved ROC plot to {args.output}")


if __name__ == '__main__':
    main()
