import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def main():
    '''
    Main function to load d0 data and plot the ROC curve.
    '''
    parser = argparse.ArgumentParser(description='Plot ROC curve from d0 data.')
    parser.add_argument('data_dir', help='Path to the folder containing signal_d0s.npy and background_d0s.npy.')
    parser.add_argument('xmin', type=float)
    parser.add_argument('xmax', type=float)
    parser.add_argument('--mode', choices=['lower', 'upper'], default='lower',
                        help="'lower': optimise minimum |d0| cut. 'upper': optimise maximum |d0| cut.")
    parser.add_argument('--significance', action='store_true', default=False,
                        help='Load d0 significance files (_sig) instead of raw d0.')
    args = parser.parse_args()

    run_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Load Data ---
    suffix = '_sig' if args.significance else ''
    signal_d0s = np.abs(np.load(os.path.join('d0_np_files', args.data_dir, f'signal_d0s{suffix}.npy')))
    background_d0s = np.abs(np.load(os.path.join('d0_np_files', args.data_dir, f'background_d0s{suffix}.npy')))

    print(f'Loaded {len(signal_d0s)} signal tracks and {len(background_d0s)} background tracks.')

    # --- ROC Curve Calculation ---
    d0_cuts = np.linspace(args.xmin, args.xmax)
    efficiencies = []
    rejections = []

    total_signal = len(signal_d0s)
    total_background = len(background_d0s)

    for cut in d0_cuts:
        if args.mode == 'lower':
            # Lower cut: keep tracks with |d0| > cut
            signal_passed = np.sum(signal_d0s > cut)
            background_rejected = np.sum(background_d0s < cut)
        else:
            # Upper cut: keep tracks with |d0| < cut
            signal_passed = np.sum(signal_d0s < cut)
            background_rejected = np.sum(background_d0s > cut)

        efficiencies.append(signal_passed / total_signal if total_signal > 0 else 0)
        rejections.append(background_rejected / total_background if total_background > 0 else 0)

    # --- Plotting ---
    cut_label = 'Lower' if args.mode == 'lower' else 'Upper'
    var_label = 'd0 significance' if args.significance else '|d0|'
    unit = '' if args.significance else ' (mm)'
    plt.figure(figsize=(10, 8))
    plt.scatter(d0_cuts, efficiencies, label='Signal Efficiency', color='blue')
    plt.scatter(d0_cuts, rejections, label='Background Rejection', color='green')
    plt.title(f'Signal Efficiency and Background Rejection vs. {cut_label} {var_label} Cut')
    plt.xlabel(f'{var_label} Cut Value{unit}')
    plt.ylabel('Fraction')
    plt.grid(True)
    plt.legend()
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(0, 1.05)
    
    best_idx = np.argmin(np.abs(np.array(efficiencies) - np.array(rejections)))
    best_cut = d0_cuts[best_idx]
    plt.axvline(x=best_cut, color='red', linestyle='--', linewidth=1)
    plt.annotate(f'Optimal {cut_label} Cut ≈ {best_cut:.3f}{unit}\n(Efficiency ≈ {efficiencies[best_idx]:.2f})',
                 xy=(best_cut, efficiencies[best_idx]),
                 xytext=(best_cut + 0.05 * (args.xmax - args.xmin), 0.5))

    sig_tag = '_sig' if args.significance else ''
    plot_filename = os.path.join('plots', f'd0_roc_{args.mode}{sig_tag}_{run_time}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f'ROC curve plot saved to {plot_filename}')

if __name__ == '__main__':
    main()