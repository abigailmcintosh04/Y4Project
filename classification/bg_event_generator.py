import subprocess
import h5py
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('n_charm', type=int)
    parser.add_argument('n_bg', type=int)
    parser.add_argument('chunk_size', type=int)
    parser.add_argument('output', type=str)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()

    runs = {'charm': args.n_charm, 'background': args.n_bg}

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
            "--process", process
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

    all_data = []

    for file_name in generated_files:
        with h5py.File(os.path.join('collisions', file_name), 'r') as f:
            all_data.append(f['events'][:])

    combined_data = np.concatenate(all_data)
    np.random.shuffle(combined_data)

    with h5py.File((os.path.join('collisions', args.output)), 'w') as f:
        # Compression
        f.create_dataset('events', data=combined_data, chunks=True)

    for file_name in generated_files:
        if os.path.exists(os.path.join('collisions', file_name)):
            os.remove(os.path.join('collisions', file_name))

    print(f'Generation of "{args.output}" complete.\nTotal events: {len(combined_data):,}')

if __name__ == '__main__':
    main()