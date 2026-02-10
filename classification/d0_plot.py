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
    parser.add_argument('d0_z0', type=str, choices=['d0', 'z0'])
    parser.add_argument('signal_file', help='Path to the .npy file with signal d0 values.')
    parser.add_argument('background_file', help='Path to the .npy file with background d0 values.')
    parser.add_argument('xmin', type=float)
    parser.add_argument('xmax', type=float)
    args = parser.parse_args()

    run_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Load Data ---
    signal_d0s = np.abs(np.load(args.signal_file))
    background_d0s = np.abs(np.load(args.background_file))

    print(f'Loaded {len(signal_d0s)} signal tracks and {len(background_d0s)} background tracks.')

    # --- ROC Curve Calculation ---
    d0_cuts = np.linspace(args.xmin, args.xmax)
    efficiencies = []
    rejections = []

    total_signal = len(signal_d0s)
    total_background = len(background_d0s)

    for cut in d0_cuts:
        # Signal efficiency: fraction of signal tracks with |d0| > cut
        signal_passed = np.sum(signal_d0s > cut)
        efficiency = signal_passed / total_signal if total_signal > 0 else 0
        efficiencies.append(efficiency)

        # Background rejection: fraction of background tracks with |d0| < cut
        background_rejected = np.sum(background_d0s < cut)
        rejection = background_rejected / total_background if total_background > 0 else 0
        rejections.append(rejection)

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    plt.scatter(d0_cuts, efficiencies, label='Signal Efficiency', color='blue')
    plt.scatter(d0_cuts, rejections, label='Background Rejection', color='green')
    if args.d0_z0 == 'd0':
        plt.title('Signal Efficiency and Background Rejection vs. d0 Cut')
        plt.xlabel('d0 Cut Value (mm)')
    elif args.d0_z0 == 'z0':
        plt.title('Signal Efficiency and Background Rejection vs. z0 Cut')
        plt.xlabel('z0 Cut Value (mm)')
    plt.ylabel('Fraction')
    plt.grid(True)
    plt.legend()
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(0, 1.05)
    
    best_idx = np.argmin(np.abs(np.array(efficiencies) - np.array(rejections)))
    best_cut = d0_cuts[best_idx]
    plt.axvline(x=best_cut, color='red', linestyle='--', linewidth=1)
    plt.annotate(f'Optimal Cut ≈ {best_cut:.3f} mm\n(Efficiency ≈ {efficiencies[best_idx]:.2f})',
                 xy=(best_cut, efficiencies[best_idx]),
                 xytext=(best_cut + 0.05 * (args.xmax - args.xmin), 0.5))

    plot_filename = os.path.join('plots', f'd0_roc_curve_{run_time}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f'ROC curve plot saved to {plot_filename}')

if __name__ == '__main__':
    main()