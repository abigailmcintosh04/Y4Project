import pythia8mc
import h5py
import numpy as np
import math
import time
import sys
import subprocess
import os
import fastjet


# PDG IDs for charm hadrons and quarks.
hadron_id_set = {411, 421, 431, 4122, -411, -421, -431, -4122}  # Charm hadrons.
quark_id_set = {4, -4} # Charm quarks.

def deltaR(eta1, phi1, eta2, phi2):
    '''Compute deltaR between two (eta,phi) points.'''
    dphi = abs(phi1 - phi2)
    if dphi > math.pi: # Handle phi wrapping around 2*pi.
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)


def deltaR_vec(eta0, phi0, etas, phis):
    '''Compute deltaR between (eta0,phi0) and arrays of etas, phis (NumPy arrays).'''
    dphi = np.abs(phis - phi0)
    # Handle phi wrapping around 2*pi in a vectorised way.
    mask = dphi > np.pi
    dphi[mask] = 2.0 * np.pi - dphi[mask]
    return np.sqrt((etas - eta0)**2 + dphi**2)


def get_c_quark_mother(particle, event, visited=None):
    '''
    Recursively traverses up the mother list of a particle to find the
    first ancestor that is a charm quark from the hard process.
    A 'visited' set is used to prevent infinite loops in complex histories.
    '''
    if visited is None:
        visited = set()

    # Prevent infinite recursion in complex event histories.
    if particle.index() in visited:
        return None
    visited.add(particle.index())

    mother_indices = particle.motherList()
    if not mother_indices:
        return None

    for mother_idx in mother_indices:
        mother = event[mother_idx]
        if mother.id() in quark_id_set:  # Base case: Found the target quark.
            return mother
        # Recursive step: Search this mother's ancestry.
        ancestor = get_c_quark_mother(mother, event, visited)
        if ancestor: # If found, pass the result up the call stack.
            return ancestor
    return None


def configure_pythia(seed=None):
    '''Configure and initialise the Pythia event generator.'''
    pythia = pythia8mc.Pythia()

    # Configure pp collisions at 13 TeV.
    pythia.readString('Beams:idA = 2212')
    pythia.readString('Beams:idB = 2212')
    pythia.readString('Beams:eCM = 13000.')

    # Enable charm quark production.
    pythia.readString('HardQCD:gg2ccbar = on')
    pythia.readString('HardQCD:qqbar2ccbar = on')

    # Set a minimum pT for the hard process to ensure "jetty" events.
    pythia.readString('PhaseSpace:pTHatMin = 20.0')

    # Enable parton showering and hadronisation.
    pythia.readString('PartonLevel:ISR = on')
    pythia.readString('PartonLevel:FSR = on')
    pythia.readString('HadronLevel:Hadronize = on')

    # Enable Multiple Parton Interactions (MPI) for a realistic underlying event.
    # pythia.readString('PartonLevel:MPI = on')

    # Quiet Pythia output.
    pythia.readString('Print:quiet = on')
    pythia.readString('Next:numberShowEvent = 0')
    pythia.readString('Next:numberShowInfo = 0')

    # Use a random seed for the random number generator.
    if seed is None:
        # Use time and process ID to ensure unique seeds across parallel shards
        # Pythia seed must be < 900,000,000
        seed = (int(time.time()) + os.getpid()) % 900_000_000
    
    print(f'Shard {os.getpid()} using Pythia seed: {seed}', flush=True)
    
    pythia.readString('Random:setSeed = on')
    pythia.readString(f'Random:seed = {seed}')

    pythia.init()
    return pythia


def single_event(event, jet_def, ptmin, consts=False, d0_cutoff=0.0):
    '''
    Process a single Pythia event to find charm hadrons and their associated jets.
    Returns a list of records for each charm hadron found in the event.
    '''
    event_records = []
    try:
        hadrons = []
        final_state_pseudojets = []

        # Create PseudoJets and find hadrons.
        for p in event:
            if p.isFinal() and p.isCharged():
                pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
                pj.set_user_index(p.index())
                final_state_pseudojets.append(pj)
            if p.id() in hadron_id_set:
                hadrons.append(p)

        # Cluster final-state particles into jets using the anti-kT algorithm.
        cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
        jets = cluster_sequence.inclusive_jets(ptmin=ptmin)

        # Process each charm hadron in the event.
        for h in hadrons:
            c_quark = get_c_quark_mother(h, event)
            if not c_quark:
                continue

            # Find the jet closest to the charm quark using a vectorised method.
            best_jet = None
            if jets:
                min_dR = 0.4
                for jet in jets:
                    dR = deltaR(c_quark.eta(), c_quark.phi(), jet.eta(), jet.phi())
                    if dR < min_dR:
                        min_dR = dR
                        best_jet = jet

            if not best_jet:
                continue

            constituents = best_jet.constituents()
            if not constituents:
                continue

            if not consts:
                # Transverse decay length of the charm hadron.
                lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)

                # Get constituent particles, excluding charm hadrons/quarks.
                particles = [event[c.user_index()] for c in constituents]
                valid_particles = [p for p in particles if p.id() not in hadron_id_set and p.id() not in quark_id_set]
                if not valid_particles:
                    continue

                px_jet, py_jet, pz_jet, e_jet = 0.0, 0.0, 0.0, 0.0
                d0_values = []
                max_pt = 0.0

                for p in valid_particles:
                    pt = p.pT()
                    if pt < 1e-9:
                        continue
                    
                    d0 = (p.xProd() * p.py() - p.yProd() * p.px()) / pt

                    if abs(d0) > d0_cutoff:
                        px_jet += p.px()
                        py_jet += p.py()
                        pz_jet += p.pz()
                        e_jet += p.e()
                        d0_values.append(d0)
                        if pt > max_pt:
                            max_pt = pt

                if len(d0_values) > 0:
                    d0_mean = np.mean(d0_values)

                    # Calculate the invariant mass of the jet.
                    jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
                    jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

                    # Calculate pT frac.
                    pt_frac = max_pt / best_jet.perp()
                else:
                    d0_mean = 0.0
                    jet_mass = 0.0
                    pt_frac = 0.0

                event_records.append((abs(h.id()), d0_mean, jet_mass, lxy, pt_frac))

            elif consts:
                return constituents, h, best_jet # Return the first valid jet found
    
    except Exception as e:
        print(f'Error processing event: {e}')

    return event_records


def generate_events(pythia, jet_def, output_file, no_events, chunk_size, dtype, ptmin, d0_cutoff=0.0):
    '''Main function to generate events and store them in an HDF5 file.'''
    start_time = time.time()
    last_report_time = start_time
    charm_events = 0

    with h5py.File(output_file, 'w') as h5file:
        # Pre-allocate the dataset to avoid slow resizing.
        dset = h5file.create_dataset(
            'events',
            shape=(no_events,),
            maxshape=(no_events,),
            dtype=dtype,
            chunks=True
        )
        write_ptr = 0
        buffer = []

        # Main generation loop: continues until the desired number of charm events is found.
        while charm_events < no_events:
            if not pythia.next():
                continue

            new_records = single_event(pythia.event, jet_def, ptmin, d0_cutoff=d0_cutoff)
            if new_records:
                buffer.extend(new_records)
                charm_events += len(new_records)

            current_time = time.time()
            if current_time - last_report_time >= 30:
                elapsed_time = current_time - start_time
                print(f'Shard {os.getpid()}: Generated {charm_events} events in {elapsed_time:.2f} seconds.')
                last_report_time = current_time

            # Flush buffer to HDF5 file when it's full or at the end of generation.
            if len(buffer) >= chunk_size or (charm_events >= no_events and buffer):
                n_to_write = len(buffer)
                if write_ptr + n_to_write > dset.shape[0]:
                    n_to_write = dset.shape[0] - write_ptr

                if n_to_write > 0:
                    arr = np.array(buffer[:n_to_write], dtype=dtype)
                    dset[write_ptr : write_ptr + n_to_write] = arr
                    write_ptr += n_to_write
                    buffer = buffer[n_to_write:]
        
        # Write any final remaining events from the buffer.
        if buffer and write_ptr < dset.shape[0]:
            n_to_write = min(len(buffer), dset.shape[0] - write_ptr)
            arr = np.array(buffer, dtype=dtype)
            dset[write_ptr : write_ptr + n_to_write] = arr
            write_ptr += n_to_write

        # Shrink dataset if fewer events were generated.
        if write_ptr < dset.shape[0]:
            dset.resize((write_ptr,))

        # The final number of events is the number of rows written.
        charm_events = write_ptr
    
    duration = time.time() - start_time
    return charm_events, duration
                

def launch_shards(script_path, args):
    '''Launch worker shards as separate processes.'''
    print(f'Master process launching {args.shards - 1} worker shards...')
    processes = []
    for i in range(1, args.shards):
        command = [
            sys.executable,
            script_path,
            args.output_file,
            str(args.no_events),
            str(args.chunk_size),
            '--shards', str(args.shards),
            '--shard-index', str(i)
        ]
        if hasattr(args, 'd0_cutoff'):
            command.extend(['--d0-cutoff', str(args.d0_cutoff)])
        p = subprocess.Popen(command) # Launch the worker process in the background.
        processes.append(p)
    
    print('All worker shards launched. Master process (shard 0) will now begin its work.')
    return processes


def merge_shards(final_output_file, temp_shard_dir, num_shards, dtype, cleanup=True):
    '''Merge shard files into a single output file.'''
    print('\nMerging shard files into a single output file...')
    shard_files = []
    total_rows = 0
    # Determine the names of all shard files and calculate the total number of events.
    base_name, ext = os.path.splitext(os.path.basename(final_output_file))
    for i in range(num_shards):
        shard_file = os.path.join(temp_shard_dir, f'{base_name}_shard_{i}{ext}')
        if os.path.exists(shard_file):
            shard_files.append(shard_file)
            with h5py.File(shard_file, 'r') as f:
                total_rows += f['events'].shape[0]

    with h5py.File(final_output_file, 'w') as final_h5:
        # Create the final dataset with the correct total size.
        final_dset = final_h5.create_dataset('events', shape=(total_rows,), dtype=dtype, chunks=True) # Changed from output_file_base to final_output_file
        write_ptr = 0
        for shard_file in shard_files:
            with h5py.File(shard_file, 'r') as f:
                data = f['events'][:]
                n_rows = data.shape[0]
                if n_rows > 0:
                    final_dset[write_ptr : write_ptr + n_rows] = data
                    write_ptr += n_rows

    print(f'Successfully merged {len(shard_files)} shard files into "{final_output_file}" with {total_rows} total events.')
    
    # Optional: Clean up temporary shard files after merging.
    if cleanup:
        print(f'Cleaning up temporary directory: {temp_shard_dir}')
        try:
            import shutil
            shutil.rmtree(temp_shard_dir)
        except OSError as e:
            print(f'Error removing directory {temp_shard_dir}: {e}')
        print('Cleanup complete.')