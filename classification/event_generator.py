import pythia8mc
import h5py
import numpy as np
import math
import argparse
import time
import sys
import subprocess
import os
import fastjet


def deltaR(eta1, phi1, eta2, phi2):
    '''Compute deltaR between two (eta,phi) points.'''
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)


def deltaR_vec(eta0, phi0, etas, phis):
    '''Compute deltaR between (eta0,phi0) and arrays of etas, phis (NumPy arrays).'''
    dphi = np.abs(phis - phi0)
    # wrap
    mask = dphi > np.pi
    dphi[mask] = 2.0 * np.pi - dphi[mask]
    return np.sqrt((etas - eta0)**2 + dphi**2)


# PDG IDs for charm hadrons and quarks.
hadron_id_set = {411, 421, 431, 4122, -411, -421, -431, -4122}  # Charm hadrons.
quark_id_set = {4, -4} # Charm quarks.


def get_c_quark_mother(particle, event, visited=None):
    '''
    Recursively traverses up the mother list of a particle to find the
    first ancestor that is a charm quark from the hard process.
    A 'visited' set is used to prevent infinite loops in complex histories.
    '''
    if visited is None:
        visited = set()

    # If we have already seen this particle in this search path, stop.
    if particle.index() in visited:
        return None
    visited.add(particle.index())

    mother_indices = particle.motherList()
    if not mother_indices:
        return None

    for mother_idx in mother_indices:
        mother = event[mother_idx]
        if mother.id() in quark_id_set: # Base case: Found the target quark.
            return mother
        # Recursive step: Search this mother's ancestry.
        ancestor = get_c_quark_mother(mother, event, visited)
        if ancestor: # If found, pass the result up the call stack.
            return ancestor
    return None


def configure_pythia():
    '''Configure and initialize the Pythia event generator.'''
    pythia = pythia8mc.Pythia()

    # Configure pp collisions at 13 TeV.
    pythia.readString('Beams:idA = 2212')
    pythia.readString('Beams:idB = 2212')
    pythia.readString('Beams:eCM = 13000.')

    # Enable charm quark production.
    pythia.readString('HardQCD:gg2ccbar = on')
    pythia.readString('HardQCD:qqbar2ccbar = on')

    # Enable parton showering and hadronization.
    pythia.readString('PartonLevel:ISR = on')
    pythia.readString('PartonLevel:FSR = on')
    pythia.readString('HadronLevel:Hadronize = on')

    # Quiet Pythia output.
    pythia.readString('Print:quiet = on')
    pythia.readString('Next:numberShowEvent = 0')
    pythia.readString('Next:numberShowInfo = 0')

    # Use a random seed for the random number generator.
    pythia.readString('Random:setSeed = on')
    pythia.readString('Random:seed = 0') # 0 means use current time

    pythia.init()
    return pythia


def single_event(event, jet_def):
    '''
    Process a single Pythia event to find charm hadrons and their associated jets.
    Returns a list of records for each charm hadron found in the event.'''
    event_records = []
    try:
        hadrons = []
        final_state_pseudojets = []

        # Create PseudoJets and find hadrons.
        for p in event:
            if p.isFinal():
                pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
                pj.set_user_index(p.index())
                final_state_pseudojets.append(pj)
            if p.id() in hadron_id_set:
                hadrons.append(p)

        # Cluster jets.
        cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
        jets = cluster_sequence.inclusive_jets(ptmin=0.0)

        # Process each charm hadron in the event.
        for h in hadrons:
            c_quark = get_c_quark_mother(h, event)
            if not c_quark:
                continue

            # Find the jet closest to the charm quark using vectorised method.
            best_jet = None
            if jets:
                jet_etas = np.array([j.eta() for j in jets])
                jet_phis = np.array([j.phi() for j in jets])
                jet_dRs = deltaR_vec(c_quark.eta(), c_quark.phi(), jet_etas, jet_phis)
                best_idx = np.argmin(jet_dRs)
                if jet_dRs[best_idx] < 0.4:
                    best_jet = jets[best_idx]

            if not best_jet:
                continue

            constituents = best_jet.constituents()
            if not constituents:
                continue

            e_jet, d0_jet, z0_jet = 0.0, 0.0, 0.0
            px_jet, py_jet, pz_jet, q_jet = 0.0, 0.0, 0.0, 0.0
            deltaR_sum = 0.0
            # Transverse decay length of the charm hadron.
            lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)
            constituent_count = 0

            # Loop over jet constituents to calculate jet properties.
            for c in constituents:
                p = event[c.user_index()]
                p_id = p.id()

                if p_id in hadron_id_set or p_id in quark_id_set:
                    continue

                deltaR_sum += deltaR(best_jet.eta(), best_jet.phi(), p.eta(), p.phi())
                e_jet += p.e()
                px_jet += p.px()
                py_jet += p.py()
                pz_jet += p.pz()
                q_jet += p.charge()

                xv, yv, zv = p.xProd(), p.yProd(), p.zProd()
                px, py, pz = p.px(), p.py(), p.pz()
                pt = math.sqrt(px**2 + py**2)

                if pt > 1e-9:
                    d0 = (xv * py - yv * px) / pt
                    d0_jet += d0

                    z0 = zv - (xv * px + yv * py) * (pz / (pt**2))
                    z0_jet += z0
                
                constituent_count += 1
            
            d0_mean = d0_jet / constituent_count if constituent_count > 0 else 0.0
            z0_mean = z0_jet / constituent_count if constituent_count > 0 else 0.0
            deltaR_mean = deltaR_sum / constituent_count if constituent_count > 0 else 0.0

            jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
            jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

            event_records.append((abs(h.id()), d0_mean, z0_mean, jet_mass, lxy, q_jet, deltaR_mean))
    except Exception as e:
        print(f'Error processing event: {e}')
    return event_records


def generate_events(pythia, jet_def, output_file, no_events, chunk_size, dtype):
    '''Main function to generate events and store them in an HDF5 file.'''
    start_time = time.time()
    charm_events = 0

    with h5py.File(output_file, 'w') as h5file:
        dset = h5file.create_dataset(
            'events',
            shape=(no_events,),
            maxshape=(no_events,),
            dtype=dtype,
            chunks=True
        )
        write_ptr = 0
        buffer = []

        while charm_events < no_events:
            if not pythia.next():
                continue

            new_records = single_event(pythia.event, jet_def)
            if new_records:
                buffer.extend(new_records)
                charm_events += len(new_records)

            # Flush buffer when it's full or at the very end.
            if len(buffer) >= chunk_size or (charm_events >= no_events and buffer):
                n_to_write = len(buffer)
                if write_ptr + n_to_write > dset.shape[0]:
                    n_to_write = dset.shape[0] - write_ptr

                if n_to_write > 0:
                    arr = np.array(buffer[:n_to_write], dtype=dtype)
                    dset[write_ptr : write_ptr + n_to_write] = arr
                    write_ptr += n_to_write
                    buffer = buffer[n_to_write:]
        
        # Write any remaining events from the buffer to the file.
        if buffer and write_ptr < dset.shape[0]:
            n_to_write = min(len(buffer), dset.shape[0] - write_ptr)
            arr = np.array(buffer, dtype=dtype)
            dset[write_ptr : write_ptr + n_to_write] = arr
            write_ptr += n_to_write

        # Shrink dataset if fewer events were generated.
        if write_ptr < dset.shape[0]:
            dset.resize((write_ptr,))

        charm_events = write_ptr
    
    duration = time.time() - start_time
    return charm_events, duration
                

def launch_shards(args):
    '''Launch worker shards as separate processes.'''
    print(f'Master process launching {args.shards - 1} worker shards...')
    processes = []
    for i in range(1, args.shards):
        command = [
            sys.executable,
            __file__,
            args.output_file,
            str(args.no_events),
            str(args.chunk_size),
            '--shards', str(args.shards),
            '--shard-index', str(i)
        ]
        p = subprocess.Popen(command)
        processes.append(p)
    
    print('All worker shards launched. Master process (shard 0) will now begin its work.')
    return processes


def merge_shards(output_file_base, num_shards, dtype, cleanup=True):
    '''Merge shard files into a single output file.'''
    print('\nMerging shard files into a single output file...')
    shard_files = []
    total_rows = 0
    base, ext = os.path.splitext(output_file_base)
    for i in range(num_shards):
        shard_file = f'{base}_shard_{i}{ext}'
        if os.path.exists(shard_file):
            shard_files.append(shard_file)
            with h5py.File(shard_file, 'r') as f:
                total_rows += f['events'].shape[0]

    with h5py.File(output_file_base, 'w') as final_h5:
        # Create the dataset with the correct total size.
        final_dset = final_h5.create_dataset('events', shape=(total_rows,), dtype=dtype, chunks=True)
        write_ptr = 0
        for shard_file in shard_files:
            with h5py.File(shard_file, 'r') as f:
                data = f['events'][:]
                n_rows = data.shape[0]
                if n_rows > 0:
                    final_dset[write_ptr : write_ptr + n_rows] = data
                    write_ptr += n_rows

    print(f'Successfully merged {len(shard_files)} shard files into "{output_file_base}" with {total_rows} total events.')
    
    if cleanup:
        print('Cleaning up temporary shard files...')
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
            except OSError as e:
                print(f'Error removing file {shard_file}: {e}')
        print('Cleanup complete.')
    
    
if __name__ == '__main__':
    # Command-line arguments for number of events and chunk size.
    parser = argparse.ArgumentParser()
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

    pythia = configure_pythia()
    events_found, duration = generate_events(pythia, jet_def, output_file, shard_events, args.chunk_size, dtype)
    print('Shard {args.shard_index}/{args.shards}: Event generation took {duration:.2f} seconds for {charm_events} events.')
    print(f'Shard {args.shard_index}/{args.shards}: Event generation took {duration:.2f} seconds for {events_found} events.')

    # If this is the master process, wait for workers and merge the results.
    if args.shards > 1 and args.shard_index == 0:
        print(f'Master process launching {args.shards - 1} worker shards...')
        print('Master process finished its work. Waiting for worker shards to complete...')
        for p in processes:
            p.wait()
        print('All shards have completed successfully.')

        merge_shards(args.output_file, args.shards, dtype, cleanup=args.cleanup)

        total_duration = time.time() - total_start_time
        print(f'\nTotal process time (generation + merge + cleanup): {total_duration:.2f} seconds.')

# no_events_total = args.no_events
# chunk_size = args.chunk_size

# # --- Parallelization Logic ---
# # If this is the master process (shard 0) and there are multiple shards,
# # launch all the other worker processes in the background.
# if args.shards > 1 and args.shard_index == 0:
#     print(f'Master process launching {args.shards - 1} worker shards...')
#     processes = []
#     for i in range(1, args.shards):
#         command = [
#             sys.executable,  # The path to the current python interpreter
#             __file__,        # The path to this script
#             args.output_file,
#             str(no_events_total),
#             str(chunk_size),
#             '--shards', str(args.shards),
#             '--shard-index', str(i)
#         ]
#         # Launch the worker process. It will run in parallel.
#         p = subprocess.Popen(command)
#         processes.append(p)
    
#     print('All worker shards launched. Master process (shard 0) will now begin its work.')

# # Calculate how many events this specific shard is responsible for.
# no_events = math.ceil(no_events_total / args.shards)

# # --- Parallelization Logic for Output File ---
# # Create a unique output filename for this shard.
# if args.shards > 1:
#     base, ext = os.path.splitext(args.output_file)
#     output_file = f'{base}_shard_{args.shard_index}{ext}'
# else:
#     output_file = args.output_file


# # Jet definition.
# jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# # Data structure for the output HDF5 file.
# dtype = np.dtype([
#     ('pdg_id_hadron', 'i4'),
#     ('d0_mean', 'f8'),
#     ('z0_mean', 'f8'),
#     ('jet_mass', 'f8'),
#     ('lxy', 'f8'),
#     ('q_jet', 'i4'),
#     ('deltaR_mean', 'f8'),
# ])

# start_time = time.time()

# # Open HDF5 file for writing.
# with h5py.File(output_file, 'w') as h5file:
#     # Pre-allocate the dataset to its full expected size to avoid slow resizing.
#     dset = h5file.create_dataset(
#         'events',
#         shape=(no_events,),
#         maxshape=(no_events,),
#         dtype=dtype,
#         chunks=True
#     )

#     write_ptr = 0 # Use a pointer to track our position in the dataset.
#     buffer = []
#     charm_events = 0

#     # Main event generation loop. Continues until the desired number of charm events is reached.
#     while charm_events < no_events:
#         if not pythia.next():
#             continue

#         hadrons = []
#         final_state_pseudojets = []
        
#         # --- Combined Particle Loop ---
#         # Create PseudoJets and find hadrons in a single pass.
#         for p in pythia.event:
#             if p.isFinal():
#                 pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
#                 pj.set_user_index(p.index())
#                 final_state_pseudojets.append(pj)
#             if p.id() in hadron_id_set:
#                 hadrons.append(p)
        
#         # Cluster jets.
#         cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
#         jets = cluster_sequence.inclusive_jets(ptmin=0.0)

#         # Process each charm hadron in the event.
#         for h in hadrons:
#             if charm_events >= no_events:
#                 break

#             c_quark = get_c_quark_mother(h, pythia.event)
#             if not c_quark:
#                 continue

#             # --- Vectorized Jet Matching ---
#             # Find the jet closest to the charm quark efficiently.
#             best_jet = None
#             if jets:
#                 jet_etas = np.array([j.eta() for j in jets])
#                 jet_phis = np.array([j.phi() for j in jets])
#                 jet_dRs = deltaR_vec(c_quark.eta(), c_quark.phi(), jet_etas, jet_phis)
#                 best_idx = np.argmin(jet_dRs)
#                 if jet_dRs[best_idx] < 0.4:
#                     best_jet = jets[best_idx]

#             if not best_jet:
#                 continue

#             constituents = best_jet.constituents()
#             if not constituents:
#                 continue

#             # (Calculations for jet properties remain the same as they were already fast)
#             e_jet, d0_jet, z0_jet = 0.0, 0.0, 0.0
#             px_jet, py_jet, pz_jet, q_jet = 0.0, 0.0, 0.0, 0.0
#             deltaR_sum = 0.0
#             # Transverse decay length of the charm hadron.
#             lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)
#             constituent_count = 0

#             # Loop over jet constituents to calculate jet properties.
#             for c in constituents:
#                 p = pythia.event[c.user_index()]
#                 p_id = p.id()

#                 if p_id in hadron_id_set or p_id in quark_id_set:
#                     continue

#                 deltaR_sum += deltaR(best_jet.eta(), best_jet.phi(), p.eta(), p.phi())
#                 e_jet += p.e()
#                 px_jet += p.px()
#                 py_jet += p.py()
#                 pz_jet += p.pz()
#                 q_jet += p.charge()

#                 xv, yv, zv = p.xProd(), p.yProd(), p.zProd()
#                 px, py, pz = p.px(), p.py(), p.pz()
#                 pt = math.sqrt(px**2 + py**2)

#                 if pt > 1e-9:
#                     d0 = (xv * py - yv * px) / pt
#                     d0_jet += d0

#                     z0 = zv - (xv * px + yv * py) * (pz / (pt**2))
#                     z0_jet += z0
                
#                 constituent_count += 1
            
#             d0_mean = d0_jet / constituent_count if constituent_count > 0 else 0.0
#             z0_mean = z0_jet / constituent_count if constituent_count > 0 else 0.0
#             deltaR_mean = deltaR_sum / constituent_count if constituent_count > 0 else 0.0

#             jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
#             jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

#             buffer.append((abs(h.id()), d0_mean, z0_mean, jet_mass, lxy, q_jet, deltaR_mean))
#             charm_events += 1

#             if constituent_count > 0:
#                 pass # This block is now only for potential future logic, the main actions are outside.

#         # --- Optimized Buffer Flushing ---
#         # Flush buffer when it's full or at the very end.
#         if len(buffer) >= chunk_size or (charm_events >= no_events and buffer):
#             n_to_write = len(buffer)
#             # Ensure we don't write past the pre-allocated space.
#             if write_ptr + n_to_write > dset.shape[0]:
#                 n_to_write = dset.shape[0] - write_ptr
            
#             arr = np.array(buffer[:n_to_write], dtype=dtype)
#             dset[write_ptr : write_ptr + n_to_write] = arr
#             write_ptr += n_to_write
#             buffer = buffer[n_to_write:] # Keep any remainder.

#     # Write any remaining events from the buffer to the file.
#     if buffer:
#         n_to_write = min(len(buffer), dset.shape[0] - write_ptr)
#         arr = np.array(buffer, dtype=dtype)
#         dset[write_ptr : write_ptr + n_to_write] = arr
#         write_ptr += n_to_write

#     # If we generated fewer events than expected, shrink the dataset to the actual size.
#     if write_ptr < dset.shape[0]:
#         dset.resize((write_ptr,))

# end_time = time.time()
# duration = end_time - start_time

# print(f'Shard {args.shard_index}/{args.shards}: Event generation took {duration:.2f} seconds for {charm_events} events.')

# # If this was the master process, wait for all worker processes to finish.
# if args.shards > 1 and args.shard_index == 0:
#     print('Master process finished its work. Waiting for worker shards to complete...')
#     for p in processes:
#         p.wait() # This will pause until the process 'p' has terminated.
    
#     print('All shards have completed successfully.')

#     # --- Final Merging Step ---
#     print('\nMerging shard files into a single output file...')
#     shard_files = []
#     total_rows = 0
#     base, ext = os.path.splitext(args.output_file)
#     for i in range(args.shards):
#         shard_file = f'{base}_shard_{i}{ext}'
#         if os.path.exists(shard_file):
#             shard_files.append(shard_file)
#             with h5py.File(shard_file, 'r') as f:
#                 total_rows += f['events'].shape[0]

#     # Create the final, merged HDF5 file.
#     with h5py.File(args.output_file, 'w') as final_h5:
#         # Create the dataset with the correct total size.
#         final_dset = final_h5.create_dataset('events', shape=(total_rows,), dtype=dtype, chunks=True)
        
#         write_ptr = 0
#         for shard_file in shard_files:
#             with h5py.File(shard_file, 'r') as f:
#                 data = f['events'][:]
#                 n_rows = data.shape[0]
#                 final_dset[write_ptr : write_ptr + n_rows] = data
#                 write_ptr += n_rows
    
#     print(f'Successfully merged {len(shard_files)} shard files into '{args.output_file}' with {total_rows} total events.')

#     # --- Optional Cleanup Step ---
#     if args.cleanup:
#         print('Cleaning up temporary shard files...')
#         for shard_file in shard_files:
#             try:
#                 os.remove(shard_file)
#             except OSError as e:
#                 print(f'Error removing file {shard_file}: {e}')
#         print('Cleanup complete.')
    
#     total_duration = time.time() - start_time
#     print(f'\nTotal process time (generation + merge + cleanup): {total_duration:.2f} seconds.')


# if __name__ == '__main__':
#     pythia = configure_pythia()