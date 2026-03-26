import numpy as np
import math
import argparse
import time
import os
import fastjet

from generator_utils import launch_shards, merge_shards, configure_pythia, generate_events

# Command-line arguments for number of events and chunk size.
parser = argparse.ArgumentParser(description='Generate particle collision events with Pythia and FastJet.')
parser.add_argument('output_file', type=str, default='collisions.h5')
parser.add_argument('no_events', type=int)
parser.add_argument('chunk_size', type=int)
parser.add_argument('--pTHatMin', type=float, default=20.0, help='Minimum pT for the hard process.')
parser.add_argument('--pTHatMax', type=float, default=None, help='Maximum pT for the hard process.')
parser.add_argument('--process', type=str, default='charm', choices=['charm', 'background'])
parser.add_argument('--shards', type=int, default=1, help='Total number of parallel shards to run.')
parser.add_argument('--shard-index', type=int, default=0, help='The index of this specific shard (0-based).')
parser.add_argument('--cleanup', action='store_true', default=True, help='Delete temporary shard files after merging.')
parser.add_argument('--d0-sig-cut', type=float, default=None, help='Minimum d0 significance (|d0/sigma|) to keep a track.')
parser.add_argument('--temp-dir', type=str, default=None, help='Directory for temporary output files (used by bg_event_generator).')
parser.add_argument('--tuning', type=str, default='monash', help='Pythia tuning to use.')
args = parser.parse_args()

total_start_time = time.time()
processes = []

# Define paths for final output and temporary shards
if args.temp_dir:
    output_base_dir = args.temp_dir
else:
    output_base_dir = 'collisions'

final_output_file = os.path.join(output_base_dir, args.output_file)

base_name, ext = os.path.splitext(args.output_file)
temp_shard_dir = os.path.join(output_base_dir, base_name)

# If this is the master process, launch all worker shards.
if args.shards > 1 and args.shard_index == 0:
    os.makedirs(temp_shard_dir, exist_ok=True)
    processes = launch_shards(__file__, args)

# Calculate events for this shard and determine its unique output filename.
shard_events = math.ceil(args.no_events / args.shards)
if args.shards > 1:
    output_file = os.path.join(temp_shard_dir, f'{base_name}_shard_{args.shard_index}{ext}')
else:
    os.makedirs(output_base_dir, exist_ok=True)
    output_file = final_output_file

# Jet definition.
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# Data structure for the output HDF5 file.
dtype = np.dtype([
    ('class_label', 'i4'),
    ('d0_mean', 'f8'),
    ('jet_mass', 'f8'),
    ('lxy', 'f8'),
    ('pt_frac', 'f8'),
    ('n_tracks', 'i4'),
    ('d0_sig_mean', 'f8'),
    ('d0_sig_max', 'f8'),
    ('jet_pt', 'f8'),
    ('d0_std', 'f8'),
    ('charge_sum', 'i4'),
])

# Configure Pythia and run the event generation for this specific shard.
pythia = configure_pythia(process=args.process, pTHatMin=args.pTHatMin, tuning=args.tuning, pTHatMax=args.pTHatMax)
events_found, duration = generate_events(pythia, jet_def, output_file, shard_events, args.chunk_size, dtype, args.pTHatMin, process=args.process, d0_sig_cut=args.d0_sig_cut)
print(f'Shard {args.shard_index}/{args.shards}: Event generation took {duration:.2f} seconds for {events_found} events.')

# If this is the master process, wait for workers and merge the results.
if args.shards > 1 and args.shard_index == 0:
    print('Master process finished its work. Waiting for worker shards to complete...')
    for p in processes:
        p.wait()
    print('All shards have completed successfully.')

    merge_shards(final_output_file, temp_shard_dir, args.shards, dtype, cleanup=args.cleanup)

    total_duration = time.time() - total_start_time
    print(f'\nTotal process time (generation + merge + cleanup): {total_duration:.2f} seconds.')
