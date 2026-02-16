import numpy as np
import fastjet
import os
from datetime import datetime
import argparse

from generator_utils import configure_pythia, single_event, hadron_id_set

parser = argparse.ArgumentParser()
parser.add_argument('no_events', type=int)
args = parser.parse_args()


def calculate_d0(particle):
    '''
    Calculate the transverse impact parameter (d0) for a particle.
    '''
    pt = particle.pT()
    if pt < 1e-9:
        return 0.0
    return (particle.xProd() * particle.py() - particle.yProd() * particle.px()) / pt


def smear_d0(true_d0, pt_gev):
    '''
    Smears the true d0 to simulate detector resolution.
    '''
    b = 0.100
    a = 0.012
    sigma = np.sqrt(a**2 + (b / pt_gev)**2)
    return np.random.normal(true_d0, sigma)


def is_signal_track(particle, hadron_index, event, visited=None):
    '''
    Recursively determine if a particle is a descendant of the charm hadron,
    checking through the ancestry chain.
    '''
    if visited is None:
        visited = set()

    # Prevent infinite recursion
    if particle.index() in visited:
        return False
    visited.add(particle.index())

    # Check immediate mothers
    mother_indices = particle.motherList()
    if not mother_indices:
        return False
    
    # If the hadron is a direct mother, it's a signal track
    if hadron_index in mother_indices:
        return True
    
    # Recursively check if any mother is a descendant
    for mother_idx in mother_indices:
        mother = event[mother_idx]
        if is_signal_track(mother, hadron_index, event, visited):
            return True
            
    return False


def main():
    '''
    Main function to generate events, analyze tracks, and save d0 data.
    '''
    run_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Configuration ---
    pythia = configure_pythia()
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    
    signal_d0s = []
    background_d0s = []

    print(f'Processing {args.no_events} events to generate ROC curve data...')

    # --- Event Loop and Data Collection ---
    for i in range(args.no_events):
        if not pythia.next():
            continue

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

            if is_signal_track(particle, hadron.index(), pythia.event):
                signal_d0s.append(abs(d0))
            else:
                background_d0s.append(abs(d0))

    print(f'Found {len(signal_d0s)} signal tracks and {len(background_d0s)} background tracks.')

    # --- Save Data ---
    signal_filename = f'signal_d0s_{run_time}.npy'
    background_filename = f'background_d0s_{run_time}.npy'
    np.save(signal_filename, np.array(signal_d0s))
    np.save(background_filename, np.array(background_d0s))
    print(f'Data saved to {signal_filename} and {background_filename}')

if __name__ == '__main__':
    main()