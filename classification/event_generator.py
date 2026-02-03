import numpy as np
import argparse
import time
import os
import multiprocessing as mp
from generator_utils import run_worker_shard, merge_shards

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Generate particle collision events with Pythia and FastJet.')
    parser.add_argument('output_file', type=str, default='collisions.h5')
    parser.add_argument('no_events', type=int, help='Total number of events to generate.')
    parser.add_argument('chunk_size', type=int, help='HDF5 chunk size.')
    parser.add_argument('--shards', type=int, default=1, help='Number of parallel CPU processes.')
    parser.add_argument('--cleanup', action='store_true', default=True, help='Delete temporary shard files after merging.')
    parser.add_argument('--d0-cutoff', type=float, default=0.0, help='Minimum d0 value to consider a track.')
    args = parser.parse_args()

    total_start_time = time.time()

    # Define paths
    collisions_dir = 'collisions'
    final_output_file = os.path.join(collisions_dir, args.output_file)
    base_name, ext = os.path.splitext(args.output_file)
    
    # Create temp directory for shards if running in parallel
    temp_shard_dir = os.path.join(collisions_dir, base_name)
    
    if not os.path.exists(collisions_dir):
        os.makedirs(collisions_dir)

    # Data structure (needed for merging)
    dtype = np.dtype([
        ('pdg_id_hadron', 'i4'), ('d0_mean', 'f8'), ('z0_mean', 'f8'),
        ('jet_mass', 'f8'), ('lxy', 'f8'), ('q_jet', 'i4'), ('deltaR_mean', 'f8'),
    ])

    if args.shards > 1:
        if not os.path.exists(temp_shard_dir):
            os.makedirs(temp_shard_dir)

        print(f"Launching {args.shards} parallel workers...")
        
        # Prepare arguments for each worker
        worker_args = []
        for i in range(args.shards):
            worker_args.append((
                i,                  # shard_index
                args.shards,        # total_shards
                temp_shard_dir,     # output_dir
                base_name,          # base_name
                ext,                # extension
                args.no_events,     # total requested events
                args.chunk_size,    # chunk size
                args.d0_cutoff      # d0 cutoff
            ))

        # Run workers using Multiprocessing Pool
        with mp.Pool(processes=args.shards) as pool:
            # starmap unpacks the argument tuples for the function
            shard_files = pool.starmap(run_worker_shard, worker_args)

        print("All workers finished. Merging results...")
        merge_shards(final_output_file, temp_shard_dir, shard_files, dtype, cleanup=args.cleanup)

    else:
        # Single process run (no overhead of temp folders/merging)
        print("Running in single-process mode.")
        run_worker_shard(0, 1, collisions_dir, base_name, ext, args.no_events, args.chunk_size, args.d0_cutoff)
        # Rename the single shard output to the final requested name
        single_shard_out = os.path.join(collisions_dir, f'{base_name}_shard_0{ext}')
        if os.path.exists(single_shard_out):
            if os.path.exists(final_output_file):
                os.remove(final_output_file)
            os.rename(single_shard_out, final_output_file)

    total_duration = time.time() - total_start_time
    print(f'\nTotal execution time: {total_duration:.2f} seconds.')