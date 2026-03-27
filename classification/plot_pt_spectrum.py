import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

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

# Class labels written by event_generator.py
# 0 = background (non-charm), 1 = other charm, 2 = Lambda_c+ signal
CLASS_SIG   = 2
CLASS_CHARM = 1
CLASS_BG    = 0


def parse_cross_sections(filepath):
    """
    Parses cross_sections.txt into a dict keyed by pt_label, e.g.:
    {
        '5-10':  {'s': 1.00637,  'b': 398.199651},
        '10-15': {'s': 0.10423,  'b': 30.570769},
        ...
    }
    """
    xs = {}
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        label = lines[i]
        s_val = float(lines[i+1].split(':')[1].strip())
        b_val = float(lines[i+2].split(':')[1].strip())
        xs[label] = {'s': s_val, 'b': b_val}
        i += 3

    return xs


# Map from pt_label to the corresponding H5 filename
FILE_MAP = {
    '5-10':  'final1m5_10.h5',
    '10-15': 'final1m10_15.h5',
    '15-20': 'final1m15_20.h5',
    '20-25': 'final1m20_25.h5',
    '25-30': 'final1m25_30.h5',
    '30+':   'final1m30_plus.h5',
}


def main():
    parser = argparse.ArgumentParser(description="Plot weighted pT spectrum across the 6 pT-sliced H5 files.")
    parser.add_argument('--cross_sections', default='cross_sections.txt',
                        help='Path to cross_sections.txt')
    parser.add_argument('--collisions_dir', default='collisions',
                        help='Directory containing the H5 files')
    parser.add_argument('--lumi_fb', type=float, default=140.0,
                        help='Luminosity in fb^-1 (default: 140)')
    parser.add_argument('--bins', type=int, default=60,
                        help='Number of histogram bins')
    parser.add_argument('--pt_min', type=float, default=5.0)
    parser.add_argument('--pt_max', type=float, default=60.0)
    parser.add_argument('--output', '-o', default='pt_spectrum.png',
                        help='Output filename')
    args = parser.parse_args()

    lumi_mb = args.lumi_fb * 1e9  # convert fb^-1 -> mb^-1

    # Parse cross sections
    if not os.path.exists(args.cross_sections):
        print(f"Error: {args.cross_sections} not found.")
        sys.exit(1)
    xs_dict = parse_cross_sections(args.cross_sections)

    bin_edges = np.linspace(args.pt_min, args.pt_max, args.bins + 1)

    # Accumulators: weighted counts per bin for each class
    H_signal    = np.zeros(args.bins)  # Lambda_c+ signal
    H_charm     = np.zeros(args.bins)  # other charm (bg charm)
    H_other     = np.zeros(args.bins)  # non-charm background

    for pt_label, filename in FILE_MAP.items():
        filepath = os.path.join(args.collisions_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue

        print(f"Loading {filename} (pT slice: {pt_label})...")
        with h5py.File(filepath, 'r') as f:
            events = f['events'][:]

        jet_pt     = events['jet_pt'].astype(float)
        class_lbl  = events['class_label']

        # Masks for each class
        mask_sig   = (class_lbl == CLASS_SIG)
        mask_charm = (class_lbl == CLASS_CHARM)
        mask_other = (class_lbl == CLASS_BG)

        n_sig_charm = np.sum(mask_sig) + np.sum(mask_charm)  # all charm events
        n_other     = np.sum(mask_other)

        sigma_s = xs_dict[pt_label]['s']
        sigma_b = xs_dict[pt_label]['b']

        # Weight = sigma * lumi / n_events_of_that_type
        w_charm = (sigma_s * lumi_mb) / n_sig_charm if n_sig_charm > 0 else 0.0
        w_other = (sigma_b * lumi_mb) / n_other     if n_other > 0     else 0.0

        print(f"  n_charm={n_sig_charm}, n_other={n_other} | w_charm={w_charm:.3e}, w_other={w_other:.3e}")

        h_sig, _   = np.histogram(jet_pt[mask_sig],   bins=bin_edges,
                                  weights=np.full(np.sum(mask_sig),   w_charm))
        h_charm, _ = np.histogram(jet_pt[mask_charm], bins=bin_edges,
                                  weights=np.full(np.sum(mask_charm), w_charm))
        h_other, _ = np.histogram(jet_pt[mask_other], bins=bin_edges,
                                  weights=np.full(np.sum(mask_other), w_other))

        H_signal += h_sig
        H_charm  += h_charm
        H_other  += h_other

    # --- Plot ---
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(9, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    samples = [H_other, H_charm, H_signal]
    colors  = ['steelblue', 'mediumseagreen', 'tomato']
    labels  = [r'Background (non-charm)', r'Charm background', r'$\Lambda_c^+$ signal']
    lws     = [1.5, 1.5, 2.0]

    for h, color, label, lw in zip(samples, colors, labels, lws):
        ax.hist(bin_centers, bins=bin_edges, weights=h,
                histtype='step', color=color, label=label, linewidth=lw)

    ax.set_yscale('log')
    ax.set_ylabel(r'Expected Event Yield  ($\mathcal{L} = ' + f'{args.lumi_fb:.1f}' + r'\ \mathrm{fb}^{-1}$)', fontsize=15)
    ax.set_title(r'Weighted Jet $p_T$ Spectrum (all $p_T$ slices combined)', fontsize=16)
    ax.set_xlim(args.pt_min, args.pt_max)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', alpha=0.3, zorder=0)
    ax.grid(True, which='minor', linestyle=':',  alpha=0.2, zorder=0)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4)
    ax.legend(loc='upper right', frameon=False)

    # --- S/B ratio panel ---
    H_bg_total = H_other + H_charm
    with np.errstate(divide='ignore', invalid='ignore'):
        sb_ratio = np.where(H_bg_total > 0, H_signal / H_bg_total, np.nan)

    ax_ratio.step(bin_edges, np.append(sb_ratio, sb_ratio[-1]),
                  where='post', color='tomato', linewidth=1.8)
    ax_ratio.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_ratio.set_ylabel(r'$S\,/\,B$', fontsize=15)
    ax_ratio.set_xlabel(r'Jet $p_T$ [GeV]', fontsize=16)
    ax_ratio.set_xlim(args.pt_min, args.pt_max)
    ax_ratio.minorticks_on()
    ax_ratio.grid(True, which='major', linestyle='--', alpha=0.3, zorder=0)
    ax_ratio.grid(True, which='minor', linestyle=':',  alpha=0.2, zorder=0)
    ax_ratio.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8)
    ax_ratio.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4)

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {args.output}")


if __name__ == '__main__':
    main()
