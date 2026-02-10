import numpy as np
import fastjet
import os
from datetime import datetime
import argparse

from generator_utils import configure_pythia, single_event, hadron_id_set, quark_id_set

parser = argparse.ArgumentParser()
parser.add_argument('no_events', type=int)
parser.add_argument('d0_z0', type=str, choices=['d0', 'z0'])
args = parser.parse_args()


def calculate_d0(particle):
    '''Calculate the transverse impact parameter (d0) for a particle.'''
    pt = particle.pT()
    if pt < 1e-9:
        return 0.0
    return (particle.xProd() * particle.py() - particle.yProd() * particle.px()) / pt


def calculate_z0(particle):
    '''Calculate the longitudinal impact parameter (z0) for a particle.'''
    pt = particle.pT()
    if pt < 1e-9:
        return 0.0
    return (particle.zProd() - (particle.xProd() * particle.px() + particle.yProd() * particle.py()) * (particle.pz() / (pt**2)))


def is_signal_track(particle, hadron):
    '''
    Determine if a particle is a "signal" track, meaning it's a direct
    daughter of the charm hadron decay.
    '''
    # Check if the particle's mother is the charm hadron.
    mother_indices = particle.motherList()
    if not mother_indices:
        return False
    
    # In Pythia, the hadron's index is what we need to match.
    return hadron.index() in mother_indices


def main():
    '''
    Main function to generate events, analyze tracks, and save d0 data.
    '''
    run_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = 'd0_np_files'
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
        result = single_event(pythia.event, jet_def, 20.0, None, None, None, consts=True)
        if not result:
            continue
        
        constituents, hadron, _ = result

        for c in constituents:
            particle = pythia.event[c.user_index()]

            # Match generator_utils.py logic: exclude quarks too
            if particle.id() in hadron_id_set or particle.id() in quark_id_set:
                continue

            if args.d0_z0 == 'd0':
                d0 = calculate_d0(particle)
            elif args.d0_z0 == 'z0':
                d0 = calculate_z0(particle)

            if is_signal_track(particle, hadron):
                signal_d0s.append(d0)
            else:
                background_d0s.append(d0)

    print(f'Found {len(signal_d0s)} signal tracks and {len(background_d0s)} background tracks.')

    # --- Save Data ---
    # --- Save Data ---
    output_subdir = os.path.join(output_dir, run_time)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    if args.d0_z0 == 'd0':
        signal_filename = os.path.join(output_subdir, 'signal_d0s.npy')
        background_filename = os.path.join(output_subdir, 'background_d0s.npy')
    elif args.d0_z0 == 'z0':
        signal_filename = os.path.join(output_subdir, 'signal_z0s.npy')
        background_filename = os.path.join(output_subdir, 'background_z0s.npy')
    np.save(signal_filename, np.array(signal_d0s))
    np.save(background_filename, np.array(background_d0s))
    print(f'Data saved to {signal_filename} and {background_filename}')

if __name__ == '__main__':
    main()