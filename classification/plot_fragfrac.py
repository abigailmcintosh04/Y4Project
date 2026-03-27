import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def parse_fragfrac(filepath):
    """
    Parses a text file with lines like:
    5-10: 7.79 pm 0.01
    10-15: 5.67 pm 0.03
    ...
    30+: 3.94 pm 0.15
    """
    bins = []
    values = []
    errors = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Split at colon
                pt_part, val_part = line.split(':')
                
                # Split value part at 'pm'
                val_str, err_str = val_part.split('pm')
                
                # Parse numeric boundaries
                pt_part = pt_part.strip()
                if '-' in pt_part:
                    low, high = pt_part.split('-')
                    bins.append((float(low), float(high)))
                elif '+' in pt_part:
                    low = pt_part.replace('+', '').strip()
                    # For a trailing + bin, we define an arbitrary upper width, e.g., 10 GeV
                    bins.append((float(low), float(low) + 20.0))
                else:
                    raise ValueError(f"Unrecognized pt bin format: {pt_part}")
                
                values.append(float(val_str.strip()))
                errors.append(float(err_str.strip()))
                
            except Exception as e:
                print(f"Error parsing line: '{line}'. Error: {e}")
                
    return bins, np.array(values), np.array(errors)

def main():
    parser = argparse.ArgumentParser(description="Plot fragmentation fraction from text file in ATLAS style.")
    parser.add_argument("input_file", help="Path to text file (e.g. fragfrac.txt)")
    parser.add_argument("--output", "-o", default="fragfrac_plot.png", help="Output plot filename")
    args = parser.parse_args()
    
    bins, values, errors = parse_fragfrac(args.input_file)
    
    if not bins:
        print("No data parsed. Exiting.")
        sys.exit(1)
        
    # Prepare histogram arrays
    bin_edges = [b[0] for b in bins] + [bins[-1][1]]
    bin_edges = np.array(bin_edges)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_widths = (bin_edges[1:] - bin_edges[:-1]) / 2.0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot as data points with error bars (crosses)
    ax.errorbar(bin_centers, values, xerr=bin_widths, yerr=errors,
                fmt='ko', markersize=6, capsize=0, elinewidth=1.5,
                label=r'Data ($\Lambda_c^+$)')
    
    # Ticks on all sides for fallback styling
    ax.tick_params(axis='both', which='major', labelsize=14, direction='in', top=True, right=True, length=8)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4)
        
    ax.set_xlabel(r'$p_T$ [GeV]', fontsize=16)
    ax.set_ylabel(r'Fragmentation Fraction $f(c \rightarrow \Lambda_c^+)$ [%]', fontsize=16)
    
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(0, max(values + errors) * 1.4)  # 40% headroom for legend/labels
    
    # Faint grid lines (both major and minor)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', alpha=0.3, zorder=0)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, zorder=0)
    
    # Legend
    # Since we manually added text at top left, we'll put the legend somewhere nice (e.g., upper right or below the text)
    ax.legend(loc='upper right', frameon=False, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved successfully to {args.output}")

if __name__ == "__main__":
    main()
