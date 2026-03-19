import subprocess
import h5py
import numpy as np
import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('n_charm', type=int)
    parser.add_argument('n_bg', type=int)
    parser.add_argument('chunk_size', type=int)
    parser.add_argument('output', type=str)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()

    collisions_dir = 'collisions'
    os.makedirs(collisions_dir, exist_ok=True)

    # Create a temporary subdirectory named after the output file (without extension).
    base_name = os.path.splitext(args.output)[0]
    temp_dir = os.path.join(collisions_dir, base_name)
    os.makedirs(temp_dir, exist_ok=True)

    runs = {'background': args.n_bg, 'charm': args.n_charm}

    generated_files = []

    for process, num_events in runs.items():
        if num_events <= 0:
            print(f"\nSkipping {process} generation (requested 0 events).")
            continue
            
        raw_file = f"raw_{process}.h5"
        generated_files.append(raw_file)
        
        print(f"Launching Pythia Generation: {process.upper()} ({num_events:,} events)")
        
        cmd = [
            "python", "event_generator.py",
            raw_file,
            str(num_events),
            str(args.chunk_size),
            "--shards", str(args.shards),
            "--process", process,
            "--temp-dir", temp_dir,
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\nSuccessfully finished generating {raw_file}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Generation failed for {process}. Exiting.")
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