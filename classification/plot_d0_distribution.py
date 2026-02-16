import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def main():
    '''
    Main function to load d0 data and plot the distribution comparison.
    '''
    parser = argparse.ArgumentParser(description='Plot d0 distribution comparison.')
    parser.add_argument('timestamp', help='Timestamp of the run (e.g. 20260210-120000) to find files in d0_np_files/.')
    parser.add_argument('--xmin', type=float, default=None, help='Minimum x-axis value')
    parser.add_argument('--xmax', type=float, default=None, help='Maximum x-axis value')
    parser.add_argument('--bins', type=int, default=100, help='Number of bins for histogram')
    parser.add_argument('--log', action='store_true', help='Use log scale for y-axis')
    args = parser.parse_args()

    run_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Load Data ---
    BASE_DIR = 'd0_np_files'
    # Look for files in the timestamped subdirectory
    # We first try to find d0 files, if not, try z0 files (or assume d0 based on script name?)
    # The prompt implies "d0 plotting script", so let's stick to d0s.
    signal_file = os.path.join(BASE_DIR, args.timestamp, 'signal_d0s.npy')
    background_file = os.path.join(BASE_DIR, args.timestamp, 'background_d0s.npy')

    # If not found, maybe check for z0s? For now, stick to d0s as per script name.
    
    try:
        signal_d0s = np.load(signal_file)
        background_d0s = np.load(background_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(f"Expected files in: {os.path.join(BASE_DIR, args.timestamp)}")
        return

    print(f'Loaded {len(signal_d0s)} signal tracks and {len(background_d0s)} background tracks.')

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Determine common range if not specified
    if args.xmin is None:
        args.xmin = min(np.min(signal_d0s), np.min(background_d0s))
    if args.xmax is None:
        args.xmax = max(np.max(signal_d0s), np.max(background_d0s))

    # Create histograms
    counts, edges, _ = plt.hist(signal_d0s, bins=args.bins, range=(args.xmin, args.xmax), alpha=0.3, 
             histtype='step', label='Charm Hadron Tracks', density=True, linewidth=2, color='blue')
    bin_centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(bin_centers, counts, linestyle='-', marker='none', color='blue')

    counts, edges, _ = plt.hist(background_d0s, bins=args.bins, range=(args.xmin, args.xmax), alpha=0.3, 
             histtype='step', label='Other Tracks', density=True, linewidth=2, color='red')
    bin_centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(bin_centers, counts, linestyle='-', marker='none', color='red')

    plt.xlabel('d0 (mm)')
    plt.ylabel('Normalized Density')
    plt.title(f'd0 Distribution Comparison ({args.timestamp})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(args.xmin, args.xmax)
    
    if args.log:
        plt.yscale('log')

    # Generate unique output filename
    base_output = os.path.join('plots', f'd0_distribution_{args.timestamp}.png')
    output_filename = base_output
    counter = 1
    while os.path.exists(output_filename):
        name, ext = os.path.splitext(base_output)
        output_filename = f"{name}_{counter}{ext}"
        counter += 1

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f'Distribution plot saved to {output_filename}')

if __name__ == '__main__':
    main()
