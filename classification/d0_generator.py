import numpy as np
import fastjet
import os
import math
import time
import sys
import subprocess
from datetime import datetime
import argparse

from generator_utils import configure_pythia, single_event, hadron_id_set


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


def launch_d0_shards(script_path, args):
    '''Launch worker shards as separate processes for d0 generation.'''
    print(f'Master process launching {args.shards - 1} worker shards...')
    processes = []
    for i in range(1, args.shards):
        command = [
            sys.executable,
            script_path,
            str(args.no_events),
            '--shards', str(args.shards),
            '--shard-index', str(i),
            '--output-dir', args.output_dir,
        ]
        p = subprocess.Popen(command)
        processes.append(p)
    
    print('All worker shards launched. Master process (shard 0) will now begin its work.')
    return processes


def merge_npy_shards(output_dir, num_shards, cleanup=True):
    '''Merge .npy shard files into single signal and background files.'''
    print('\nMerging shard files into single output files...')
    all_signal = []
    all_background = []

    for i in range(num_shards):
        signal_file = os.path.join(output_dir, f'signal_d0s_shard_{i}.npy')
        background_file = os.path.join(output_dir, f'background_d0s_shard_{i}.npy')

        if os.path.exists(signal_file):
            all_signal.append(np.load(signal_file))
        if os.path.exists(background_file):
            all_background.append(np.load(background_file))

    merged_signal = np.concatenate(all_signal) if all_signal else np.array([])
    merged_background = np.concatenate(all_background) if all_background else np.array([])

    signal_out = os.path.join(output_dir, 'signal_d0s.npy')
    background_out = os.path.join(output_dir, 'background_d0s.npy')
    np.save(signal_out, merged_signal)
    np.save(background_out, merged_background)

    print(f'Successfully merged {num_shards} shards: '
          f'{len(merged_signal)} signal tracks, {len(merged_background)} background tracks.')
    print(f'Data saved to {signal_out} and {background_out}')

    # Clean up temporary shard files
    if cleanup:
        print('Cleaning up temporary shard files...')
        for i in range(num_shards):
            for fn in [f'signal_d0s_shard_{i}.npy', f'background_d0s_shard_{i}.npy']:
                path = os.path.join(output_dir, fn)
                if os.path.exists(path):
                    os.remove(path)
        print('Cleanup complete.')


def main():
    '''
    Main function to generate events, analyze tracks, and save d0 data.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('no_events', type=int)
    parser.add_argument('--shards', type=int, default=1, help='Total number of parallel shards to run.')
    parser.add_argument('--shard-index', type=int, default=0, help='The index of this specific shard (0-based).')
    parser.add_argument('--cleanup', action='store_true', default=True, help='Delete temporary shard files after merging.')
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for .npy files (set automatically for shards).')
    args = parser.parse_args()

    total_start_time = time.time()
    processes = []

    # Set up output directory
    if not args.output_dir:
        run_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        args.output_dir = os.path.join('d0_np_files', run_time)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # If this is the master process, launch all worker shards.
    if args.shards > 1 and args.shard_index == 0:
        processes = launch_d0_shards(__file__, args)

    # Calculate events for this shard.
    shard_events = math.ceil(args.no_events / args.shards)

    # --- Configuration ---
    pythia = configure_pythia()
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    
    signal_d0s = []
    background_d0s = []

    print(f'Shard {args.shard_index}/{args.shards}: Processing {shard_events} events to generate ROC curve data...')

    # --- Event Loop and Data Collection ---
    charm_events = 0
    while charm_events < shard_events:
        if not pythia.next():
            continue

        # We set consts=True to get the first valid jet and its constituents
        try:
            constituents, h, best_jet = single_event(pythia.event, jet_def, ptmin=20.0, consts=True)
        except ValueError:
            continue
        
        for i, c in enumerate(constituents):
            p = pythia.event[c.user_index()]
            p_id = p.id()

            # We don't want to evaluate the charm hadron itself
            if p_id in hadron_id_set:
                continue

            d0 = calculate_d0(p)
            d0 = smear_d0(d0, p.pT())

            if is_signal_track(p, h.index(), pythia.event):
                signal_d0s.append(d0)
            else:
                background_d0s.append(d0)
        charm_events += 1

    duration = time.time() - total_start_time
    print(f'Shard {args.shard_index}/{args.shards}: Found {len(signal_d0s)} signal tracks and '
          f'{len(background_d0s)} background tracks in {duration:.2f} seconds.')

    # --- Save Data ---
    if args.shards > 1:
        signal_filename = os.path.join(args.output_dir, f'signal_d0s_shard_{args.shard_index}.npy')
        background_filename = os.path.join(args.output_dir, f'background_d0s_shard_{args.shard_index}.npy')
    else:
        signal_filename = os.path.join(args.output_dir, 'signal_d0s.npy')
        background_filename = os.path.join(args.output_dir, 'background_d0s.npy')

    np.save(signal_filename, np.array(signal_d0s))
    np.save(background_filename, np.array(background_d0s))
    print(f'Shard {args.shard_index} data saved to {signal_filename} and {background_filename}')

    # If this is the master process, wait for workers and merge the results.
    if args.shards > 1 and args.shard_index == 0:
        print('Master process finished its work. Waiting for worker shards to complete...')
        for p in processes:
            p.wait()
        print('All shards have completed successfully.')

        merge_npy_shards(args.output_dir, args.shards, cleanup=args.cleanup)

        total_duration = time.time() - total_start_time
        print(f'\nTotal process time (generation + merge + cleanup): {total_duration:.2f} seconds.')


if __name__ == '__main__':
    main()