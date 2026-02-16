import numpy as np
import math
import argparse
import time
import os
import fastjet

from generator_utils import launch_shards, merge_shards, configure_pythia, generate_events

D0_LOW = 0.04
D0_HIGH = 0.38

# Command-line arguments for number of events and chunk size.
parser = argparse.ArgumentParser(description='Generate particle collision events with Pythia and FastJet.')
parser.add_argument('output_file', type=str, default='collisions.h5')
parser.add_argument('no_events', type=int)
parser.add_argument('chunk_size', type=int)
parser.add_argument('--shards', type=int, default=1, help='Total number of parallel shards to run.')
parser.add_argument('--shard-index', type=int, default=0, help='The index of this specific shard (0-based).')
parser.add_argument('--cleanup', action='store_true', default=True, help='Delete temporary shard files after merging.')
parser.add_argument('--d0-low', type=float, default=D0_LOW, help='Minimum d0 value to consider a track.')
parser.add_argument('--d0-high', type=float, default=D0_HIGH, help='Maximum d0 value to consider a track.')
args = parser.parse_args()

total_start_time = time.time()
processes = []

# Define paths for final output and temporary shards
collisions_dir = 'collisions'
final_output_file = os.path.join(collisions_dir, args.output_file)

base_name, ext = os.path.splitext(args.output_file)
temp_shard_dir = os.path.join(collisions_dir, base_name)

# If this is the master process, launch all worker shards.
if args.shards > 1 and args.shard_index == 0:
    if not os.path.exists(temp_shard_dir):
        os.makedirs(temp_shard_dir)
    processes = launch_shards(__file__, args)

# Calculate events for this shard and determine its unique output filename.
shard_events = math.ceil(args.no_events / args.shards)
if args.shards > 1:
    output_file = os.path.join(temp_shard_dir, f'{base_name}_shard_{args.shard_index}{ext}')
else:
    if not os.path.exists(collisions_dir):
        os.makedirs(collisions_dir)
    output_file = final_output_file

# Jet definition.
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# Data structure for the output HDF5 file.
dtype = np.dtype([
    ('pdg_id_hadron', 'i4'),
    ('d0_mean', 'f8'),
    ('jet_mass', 'f8'),
    ('lxy', 'f8'),
    ('pt_frac', 'f8'),
])

# Configure Pythia and run the event generation for this specific shard.
pythia = configure_pythia()
events_found, duration = generate_events(pythia, jet_def, output_file, shard_events, args.chunk_size, dtype, 20.0, d0_low=args.d0_low, d0_high=args.d0_high)
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
