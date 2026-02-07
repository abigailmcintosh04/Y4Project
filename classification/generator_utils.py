import pythia8mc
import h5py
import numpy as np
import math
import time
import sys
import os
import fastjet

# PDG IDs for charm hadrons and quarks.
hadron_id_set = {411, 421, 431, 4122, -411, -421, -431, -4122}  # Charm hadrons.
quark_id_set = {4, -4} # Charm quarks.

def deltaR(eta1, phi1, eta2, phi2):
    '''Compute deltaR between two (eta,phi) points.'''
    dphi = abs(phi1 - phi2)
    if dphi > math.pi: 
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)


def get_c_quark_mother(particle, event, visited=None):
    '''
    Recursively traverses up the mother list of a particle to find the
    first ancestor that is a charm quark from the hard process.
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
        if ancestor: 
            return ancestor
    return None


def configure_pythia(seed=0):
    '''
    Configure and initialise the Pythia event generator.
    Args:
        seed (int): Random number seed. Essential for parallel processing.
    '''
    pythia = pythia8mc.Pythia()

    # Configure pp collisions at 13 TeV.
    pythia.readString('Beams:idA = 2212')
    pythia.readString('Beams:idB = 2212')
    pythia.readString('Beams:eCM = 13000.')

    # Enable charm quark production.
    pythia.readString('HardQCD:gg2ccbar = on')
    pythia.readString('HardQCD:qqbar2ccbar = on')

    # Set a minimum pT for the hard process.
    pythia.readString('PhaseSpace:pTHatMin = 20.0')

    # Enable parton showering and hadronisation.
    pythia.readString('PartonLevel:ISR = on')
    pythia.readString('PartonLevel:FSR = on')
    pythia.readString('HadronLevel:Hadronize = on')

    # Quiet Pythia output.
    pythia.readString('Print:quiet = on')
    pythia.readString('Next:numberShowEvent = 0')
    pythia.readString('Next:numberShowInfo = 0')

    # Use a specific random seed (critical for multiprocessing)
    pythia.readString('Random:setSeed = on')
    pythia.readString(f'Random:seed = {seed}')

    pythia.init()
    return pythia


def single_event(event, jet_def, jet_ptmin, d0_min, d0_max, track_pt_min, consts=False):
    '''
    Process a single Pythia event to find charm hadrons and their associated jets.
    Applies experimental cuts to select valid tracks for jet variables.
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
        jets = cluster_sequence.inclusive_jets(ptmin=jet_ptmin)

        # Process each charm hadron in the event.
        for h in hadrons:
            c_quark = get_c_quark_mother(h, event)
            if not c_quark:
                continue

            # Find the jet closest to the charm quark.
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
                # Transverse decay length of the charm hadron (Truth info for reference).
                lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)

                # Get constituent particles, excluding the charm hadrons/quarks themselves.
                particles = [event[c.user_index()] for c in constituents]
                valid_particles = [p for p in particles if p.id() not in hadron_id_set and p.id() not in quark_id_set]
                
                if not valid_particles:
                    continue

                px_jet, py_jet, pz_jet, e_jet = 0.0, 0.0, 0.0, 0.0
                d0_values = []


                # --- TRACK SELECTION LOOP ---
                for p in valid_particles:
                    pt = p.pT()
                    
                    # 1. Kinematic Cut: fast tracks only
                    if pt < track_pt_min:
                        continue
                    
                    # Calculate impact parameters
                    d0 = (p.xProd() * p.py() - p.yProd() * p.px()) / pt
                    z0 = p.zProd() - (p.xProd() * p.px() + p.yProd() * p.py()) * (p.pz() / (pt**2))
                    abs_d0 = abs(d0)

                    # 2. Geometric Band-Pass Filter
                    # d0_min removes Prompt tracks (too close)
                    # d0_max removes Strange hadrons / Material interactions (too far)
                    if abs_d0 > d0_min and abs_d0 < d0_max:
                        px_jet += p.px()
                        py_jet += p.py()
                        pz_jet += p.pz()
                        e_jet += p.e()
                        d0_values.append(d0)

                # Only save the event if we found tracks passing the strict cuts
                if len(d0_values) > 0:
                    d0_mean = np.mean(d0_values)

                    # Calculate the invariant mass of the jet using ONLY selected tracks
                    jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
                    jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0
                    
                    # Calculate fractional pT
                    pt_frac = math.sqrt(px_jet**2 + py_jet**2) / best_jet.perp()

                    event_records.append((abs(h.id()), d0_mean, jet_mass, lxy, pt_frac))

            elif consts:
                return constituents, h, best_jet
    
    except Exception as e:
        print(f'Error processing event: {e}')

    return event_records


def generate_events(pythia, jet_def, output_file, no_events, chunk_size, dtype, jet_ptmin, d0_min, d0_max, track_pt_min):
    '''Main loop to generate events and store them in an HDF5 file.'''
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

        # Main generation loop
        while charm_events < no_events:
            if not pythia.next():
                continue

            new_records = single_event(
                pythia.event, 
                jet_def, 
                jet_ptmin=jet_ptmin, 
                d0_min=d0_min, 
                d0_max=d0_max, 
                track_pt_min=track_pt_min
            )

            if new_records:
                buffer.extend(new_records)
                charm_events += len(new_records)

            current_time = time.time()
            if current_time - last_report_time >= 30:
                elapsed_time = current_time - start_time
                print(f'PID {os.getpid()}: Generated {charm_events}/{no_events} events ({elapsed_time:.0f}s)')
                last_report_time = current_time

            # Flush buffer to HDF5 file when it's full
            if len(buffer) >= chunk_size or (charm_events >= no_events and buffer):
                n_to_write = len(buffer)
                if write_ptr + n_to_write > dset.shape[0]:
                    n_to_write = dset.shape[0] - write_ptr

                if n_to_write > 0:
                    arr = np.array(buffer[:n_to_write], dtype=dtype)
                    dset[write_ptr : write_ptr + n_to_write] = arr
                    write_ptr += n_to_write
                    buffer = buffer[n_to_write:]
        
        # Write any final remaining events
        if buffer and write_ptr < dset.shape[0]:
            n_to_write = min(len(buffer), dset.shape[0] - write_ptr)
            arr = np.array(buffer, dtype=dtype)
            dset[write_ptr : write_ptr + n_to_write] = arr
            write_ptr += n_to_write

        # Shrink dataset if fewer events were generated
        if write_ptr < dset.shape[0]:
            dset.resize((write_ptr,))

        charm_events = write_ptr
    
    duration = time.time() - start_time
    return charm_events, duration


def run_worker_shard(shard_index, total_shards, output_dir, base_name, ext, no_events, chunk_size, d0_min, d0_max, track_pt_min):
    '''
    Worker function to be run by multiprocessing.Pool.
    Initializes its own Pythia instance with a unique seed.
    '''
    # Unique seed for each shard
    seed = (int(time.time()) % 100000) * 100 + shard_index
    
    output_file = os.path.join(output_dir, f'{base_name}_shard_{shard_index}{ext}')
    
    # Calculate events for this specific shard
    shard_events = math.ceil(no_events / total_shards)

    # Output structure
    # Output structure
    dtype = np.dtype([
        ('pdg_id_hadron', 'i4'), 
        ('d0_mean', 'f8'), 
        ('jet_mass', 'f8'), 
        ('lxy', 'f8'), 
        ('pt_frac', 'f8'),
    ])

    # Configure
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    pythia = configure_pythia(seed=seed)

    print(f'Shard {shard_index} started. Target: {shard_events} events.')
    
    # Pass 20.0 as the jet_ptmin (hardcoded here as standard analysis cut)
    events_found, duration = generate_events(
        pythia, 
        jet_def, 
        output_file, 
        shard_events, 
        chunk_size, 
        dtype, 
        jet_ptmin=20.0, 
        d0_min=d0_min, 
        d0_max=d0_max, 
        track_pt_min=track_pt_min
    )
    
    print(f'Shard {shard_index} finished: {events_found} events in {duration:.2f}s')
    return output_file


def merge_shards(final_output_file, temp_shard_dir, shard_files, dtype, cleanup=True):
    '''Merge shard files into a single output file.'''
    print(f'\nMerging {len(shard_files)} shard files...')
    
    total_rows = 0
    valid_files = []
    
    # First pass: calculate total size
    for f_path in shard_files:
        if os.path.exists(f_path):
            valid_files.append(f_path)
            with h5py.File(f_path, 'r') as f:
                if 'events' in f:
                    total_rows += f['events'].shape[0]

    with h5py.File(final_output_file, 'w') as final_h5:
        # Create the final dataset
        final_dset = final_h5.create_dataset('events', shape=(total_rows,), dtype=dtype, chunks=True)
        write_ptr = 0
        for f_path in valid_files:
            with h5py.File(f_path, 'r') as f:
                if 'events' in f:
                    data = f['events'][:]
                    n_rows = data.shape[0]
                    if n_rows > 0:
                        final_dset[write_ptr : write_ptr + n_rows] = data
                        write_ptr += n_rows

    print(f'Merged output saved to "{final_output_file}" ({total_rows} events).')
    
    if cleanup:
        print(f'Cleaning up temporary directory: {temp_shard_dir}')
        import shutil
        try:
            shutil.rmtree(temp_shard_dir)
        except OSError as e:
            print(f'Error cleaning up: {e}')