import numpy as np
import matplotlib.pyplot as plt
import fastjet
import math
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('xmin', type=float)
parser.add_argument('xmax', type=float)
args = parser.parse_args()

from generator_utils import configure_pythia, single_event, hadron_id_set

def calculate_d0(particle):
    """Calculate the transverse impact parameter (d0) for a particle."""
    pt = particle.pT()
    if pt < 1e-9:
        return 0.0
    return (particle.xProd() * particle.py() - particle.yProd() * particle.px()) / pt

def is_signal_track(particle, hadron):
    """
    Determine if a particle is a "signal" track, meaning it's a direct
    daughter of the charm hadron decay.
    """
    # Check if the particle's mother is the charm hadron.
    mother_indices = particle.motherList()
    if not mother_indices:
        return False
    
    # In Pythia, the hadron's index is what we need to match.
    return hadron.index() in mother_indices


def main():
    """
    Main function to generate events, analyze tracks, and plot the ROC curve.
    """
    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Configuration ---
    pythia = configure_pythia()
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    num_events_to_process = 5000
    
    signal_d0s = []
    background_d0s = []

    print(f"Processing {num_events_to_process} events to generate ROC curve data...")

    # --- Event Loop and Data Collection ---
    for i in range(num_events_to_process):
        if not pythia.next():
            continue

        # Use a modified single_event logic to get constituents and the hadron
        # We set consts=True to get the first valid jet and its constituents
        result = single_event(pythia.event, jet_def, ptmin=20.0, consts=True)
        if not result:
            continue
        
        constituents, hadron, _ = result

        for c in constituents:
            particle = pythia.event[c.user_index()]

            # We don't want to evaluate the charm hadron itself
            if particle.id() in hadron_id_set:
                continue

            d0 = calculate_d0(particle)

            if is_signal_track(particle, hadron):
                signal_d0s.append(abs(d0))
            else:
                background_d0s.append(abs(d0))

    print(f"Found {len(signal_d0s)} signal tracks and {len(background_d0s)} background tracks.")

    # --- ROC Curve Calculation ---
    d0_cuts = np.linspace(0, 1.0, 200) # Test d0 cuts from 0 to 1.0 mm
    efficiencies = []
    rejections = []

    total_signal = len(signal_d0s)
    total_background = len(background_d0s)

    for cut in d0_cuts:
        # Signal efficiency: fraction of signal tracks with |d0| > cut
        signal_passed = np.sum(np.array(signal_d0s) > cut)
        efficiency = signal_passed / total_signal if total_signal > 0 else 0
        efficiencies.append(efficiency)

        # Background rejection: fraction of background tracks with |d0| < cut
        background_rejected = np.sum(np.array(background_d0s) < cut)
        rejection = background_rejected / total_background if total_background > 0 else 0
        rejections.append(rejection)

    # --- Plotting ---
    plt.figure(figsize=(10, 7))
    plt.plot(d0_cuts, efficiencies, label='Signal Efficiency', color='blue')
    plt.plot(d0_cuts, rejections, label='Background Rejection', color='green')
    
    plt.xlabel('d0 Cut Value (mm)')
    plt.ylabel('Fraction')
    plt.title('Signal Efficiency and Background Rejection vs. d0 Cut')
    plt.grid(True)
    plt.legend()
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(0, 1.05)
    
    # Find and annotate the intersection point of the two curves
    best_idx = np.argmin(np.abs(np.array(efficiencies) - np.array(rejections)))
    best_cut = d0_cuts[best_idx]
    plt.axvline(x=best_cut, color='red', linestyle='--', linewidth=1)
    plt.annotate(f'Optimal Cut ≈ {best_cut:.3f} mm\n(Efficiency ≈ {efficiencies[best_idx]:.2f})',
                 xy=(best_cut, efficiencies[best_idx]),
                 xytext=(best_cut + 0.02, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plot_filename = os.path.join('plots', f'd0_roc_curve_{run_time}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve plot saved to '{plot_filename}'")

if __name__ == '__main__':
    main()