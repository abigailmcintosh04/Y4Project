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
parser.add_argument('--shards', type=int, default=1, help='Total number of parallel shards to run.')
parser.add_argument('--shard-index', type=int, default=0, help='The index of this specific shard (0-based).')
parser.add_argument('--cleanup', action='store_true', default=True, help='Delete temporary shard files after merging.')
args = parser.parse_args()

total_start_time = time.time()
processes = []

# If this is the master process, launch all worker shards.
if args.shards > 1 and args.shard_index == 0:
    processes = launch_shards(args)

# Calculate events for this shard and determine its unique output filename.
shard_events = math.ceil(args.no_events / args.shards)
if args.shards > 1:
    base, ext = os.path.splitext(args.output_file)
    output_file = f'{base}_shard_{args.shard_index}{ext}'
else:
    output_file = args.output_file

# Jet definition.
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# Data structure for the output HDF5 file.
dtype = np.dtype([
    ('pdg_id_hadron', 'i4'),
    ('d0_mean', 'f8'),
    ('z0_mean', 'f8'),
    ('jet_mass', 'f8'),
    ('lxy', 'f8'),
    ('q_jet', 'i4'),
    ('deltaR_mean', 'f8'),
])

# Configure Pythia and run the event generation for this specific shard.
pythia = configure_pythia()
events_found, duration = generate_events(pythia, jet_def, output_file, shard_events, args.chunk_size, dtype)
print(f'Shard {args.shard_index}/{args.shards}: Event generation took {duration:.2f} seconds for {events_found} events.')

# If this is the master process, wait for workers and merge the results.
if args.shards > 1 and args.shard_index == 0:
    print('Master process finished its work. Waiting for worker shards to complete...')
    for p in processes:
        p.wait()
    print('All shards have completed successfully.')

    merge_shards(args.output_file, args.shards, dtype, cleanup=args.cleanup)

    total_duration = time.time() - total_start_time
    print(f'\nTotal process time (generation + merge + cleanup): {total_duration:.2f} seconds.')
