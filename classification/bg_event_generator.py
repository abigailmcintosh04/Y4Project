import subprocess
import h5py
import numpy as np
import os
import shutil
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('n_charm', type=int)
    parser.add_argument('n_bg', type=int)
    parser.add_argument('chunk_size', type=int)
    parser.add_argument('output', type=str)
    parser.add_argument('--charm-shards', type=int, default=1, help='Number of shards for charm generation.')
    parser.add_argument('--bg-shards', type=int, default=1, help='Number of shards for background generation.')
    parser.add_argument('--pTHatMin', type=float, default=20.0, help='Minimum pT for the hard process.')
    parser.add_argument('--pTHatMax', type=float, default=None, help='Maximum pT for the hard process.')
    parser.add_argument('--d0-sig-cut', type=float, default=None, help='Minimum d0 significance (|d0/sigma|) to keep a track.')
    parser.add_argument('--tuning', type=str, default='monash', help='Pythia tuning to use.')
    args = parser.parse_args()

    collisions_dir = 'collisions'
    os.makedirs(collisions_dir, exist_ok=True)

    # Create a temporary subdirectory named after the output file (without extension).
    base_name = os.path.splitext(args.output)[0]
    temp_dir = os.path.join(collisions_dir, base_name)
    os.makedirs(temp_dir, exist_ok=True)

    runs = {
        'background': (args.n_bg, args.bg_shards),
        'charm': (args.n_charm, args.charm_shards),
    }

    # Launch all processes in parallel.
    generated_files = []
    active_processes = []

    for process, (num_events, shards) in runs.items():
        if num_events <= 0:
            print(f"Skipping {process} generation (requested 0 events).")
            continue

        raw_file = f"raw_{process}.h5"
        generated_files.append(raw_file)

        print(f"Launching Pythia Generation: {process.upper()} ({num_events:,} events, {shards} shards)")

        cmd = [
            "python", "event_generator.py",
            raw_file,
            str(num_events),
            str(args.chunk_size),
            "--shards", str(shards),
            "--process", process,
            "--temp-dir", temp_dir,
            "--pTHatMin", str(args.pTHatMin),
            "--tuning", str(args.tuning),
        ]
        if args.pTHatMax is not None:
            cmd.extend(["--pTHatMax", str(args.pTHatMax)])
        if args.d0_sig_cut is not None:
            cmd.extend(["--d0-sig-cut", str(args.d0_sig_cut)])

        p = subprocess.Popen(cmd)
        active_processes.append((process, p))

    # Wait for all processes to complete.
    failed = False
    for process, p in active_processes:
        p.wait()
        if p.returncode != 0:
            print(f"\n[ERROR] Generation failed for {process} (exit code {p.returncode}).")
            failed = True
        else:
            print(f"Successfully finished generating {process}.")

    if failed:
        print("One or more generation processes failed. Exiting.")
        exit(1)

    if not generated_files:
        print("\nNo events generated. Exiting.")
        return

    # Merge all generated files into a single output file in collisions/.
    all_data = []

    for file_name in generated_files:
        file_path = os.path.join(temp_dir, file_name)
        with h5py.File(file_path, 'r') as f:
            all_data.append(f['events'][:])

    combined_data = np.concatenate(all_data)
    np.random.shuffle(combined_data)

    final_output = os.path.join(collisions_dir, args.output)
    with h5py.File(final_output, 'w') as f:
        f.create_dataset('events', data=combined_data, chunks=True)

    # Clean up the temporary directory.
    shutil.rmtree(temp_dir)

    print(f'Generation of "{args.output}" complete.\nTotal events: {len(combined_data):,}')

if __name__ == '__main__':
    main()