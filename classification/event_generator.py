import numpy as np
import argparse
import time
import os
import multiprocessing as mp
from generator_utils import run_worker_shard, merge_shards

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Generate particle collision events with Pythia and FastJet.')
    parser.add_argument('output_file', type=str, default='collisions.h5', help='Output HDF5 filename.')
    parser.add_argument('no_events', type=int, help='Total number of events to generate.')
    parser.add_argument('chunk_size', type=int, help='HDF5 chunk size (recommended: 10000).')
    
    # Parallelization arguments
    parser.add_argument('--shards', type=int, default=1, help='Number of parallel CPU processes.')
    parser.add_argument('--cleanup', action='store_true', default=True, help='Delete temporary shard files after merging.')

    # Physics Cuts (Experimental Track Selection)
    parser.add_argument('--track-pt-min', type=float, default=None, help='Minimum pT for tracks in GeV (default: None).')
    parser.add_argument('--d0-min', type=float, default=None, help='Minimum d0 in mm to reject prompt tracks (default: None).')
    parser.add_argument('--d0-max', type=float, default=None, help='Maximum d0 in mm to reject Strange hadrons (default: None).')

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
        ('pdg_id_hadron', 'i4'), 
        ('d0_mean', 'f8'), 
        ('jet_mass', 'f8'), 
        ('lxy', 'f8'), 
        ('pt_frac', 'f8'),
    ])

    # 1. PARALLEL MODE
    if args.shards > 1:
        if not os.path.exists(temp_shard_dir):
            os.makedirs(temp_shard_dir)

        print(f"Launching {args.shards} parallel workers...")
        
        cuts_info = []
        if args.track_pt_min is not None: cuts_info.append(f"pT > {args.track_pt_min} GeV")
        if args.d0_min is not None: cuts_info.append(f"|d0| > {args.d0_min} mm")
        if args.d0_max is not None: cuts_info.append(f"|d0| < {args.d0_max} mm")
        
        print(f"Cuts: {', '.join(cuts_info) if cuts_info else 'None'}")
        
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
                args.d0_min,        # d0_min
                args.d0_max,        # d0_max
                args.track_pt_min   # track_pt_min
            ))

        # Run workers using Multiprocessing Pool
        with mp.Pool(processes=args.shards) as pool:
            # starmap unpacks the argument tuples for the function
            shard_files = pool.starmap(run_worker_shard, worker_args)

        print("All workers finished. Merging results...")
        merge_shards(final_output_file, temp_shard_dir, shard_files, dtype, cleanup=args.cleanup)

    # 2. SINGLE PROCESS MODE
    else:
        print("Running in single-process mode.")
        
        cuts_info = []
        if args.track_pt_min is not None: cuts_info.append(f"pT > {args.track_pt_min} GeV")
        if args.d0_min is not None: cuts_info.append(f"|d0| > {args.d0_min} mm")
        if args.d0_max is not None: cuts_info.append(f"|d0| < {args.d0_max} mm")
        
        print(f"Cuts: {', '.join(cuts_info) if cuts_info else 'None'}")
        
        # Run directly in this process
        run_worker_shard(
            0, 
            1, 
            collisions_dir, 
            base_name, 
            ext, 
            args.no_events, 
            args.chunk_size, 
            args.d0_min, 
            args.d0_max, 
            args.track_pt_min
        )
        
        # Rename the single shard output to the final requested name
        single_shard_out = os.path.join(collisions_dir, f'{base_name}_shard_0{ext}')
        if os.path.exists(single_shard_out):
            if os.path.exists(final_output_file):
                os.remove(final_output_file)
            os.rename(single_shard_out, final_output_file)

    total_duration = time.time() - total_start_time
    print(f'\nTotal execution time: {total_duration:.2f} seconds.')