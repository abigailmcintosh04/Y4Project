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


def get_c_hadron_daughter(particle, event, visited=None):
    '''
    Recursively traverses down the daughter list of a particle (e.g., a charm
    quark) to find the first descendant that is a charm hadron.
    A 'visited' set is used to prevent infinite loops.
    '''
    if visited is None:
        visited = set()

    # Base case: If the particle itself is a charm hadron, we've found it.
    if particle.id() in hadron_id_set:
        return particle

    if particle.index() in visited:
        return None
    visited.add(particle.index())

    daughter_indices = particle.daughterList()
    if not daughter_indices:
        return None

    for daughter_idx in daughter_indices:
        daughter = event[daughter_idx]
        # Base case: Found a charm hadron.
        if daughter.id() in hadron_id_set:
            return daughter
        # Recursive step: Search this daughter's descendants.
        descendant = get_c_hadron_daughter(daughter, event, visited)
        if descendant: # If found, pass the result up the call stack.
            return descendant
    return None


def configure_pythia():
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
    pythia.readString('PartonLevel:MPI = on')

    # Quiet Pythia output.
    pythia.readString('Print:quiet = on')
    pythia.readString('Next:numberShowEvent = 0')
    pythia.readString('Next:numberShowInfo = 0')

    # Use a random seed for the random number generator.
    pythia.readString('Random:setSeed = on')
    pythia.readString('Random:seed = 0') # 0 means use current time

    pythia.init()
    return pythia


def single_event(event, jet_def, consts=False):
    '''
    Process a single Pythia event to find charm hadrons and their associated jets.
    Returns a list of records for each charm hadron found in the event.'''
    event_records = []
    quarks = []
    try:
        final_state_pseudojets = []

        # Create PseudoJets for jet clustering and find initial charm quarks.
        for p in event:
            if p.isFinal():
                pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
                pj.set_user_index(p.index())
                final_state_pseudojets.append(pj)
            if p.id() in quark_id_set and not p.motherList():
                quarks.append(p)

        # Cluster final-state particles into jets using the anti-kT algorithm.
        cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
        jets = cluster_sequence.inclusive_jets(ptmin=5.0)

        # For each charm quark, find its corresponding hadron and jet.
        for c_quark in quarks:
            h = get_c_hadron_daughter(c_quark, event)
            if not h:
                continue

            # Find the jet closest to the charm quark using a vectorised method.
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

            if not consts:
                # Transverse decay length of the charm hadron.
                lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)

                # Vectorised calculation of jet properties.
                p_indices = [c.user_index() for c in constituents]
                particles = [event[i] for i in p_indices]

                # Filter out charm quarks/hadrons from property calculations.
                filtered_particles = [p for p in particles if p.id() not in hadron_id_set and p.id() not in quark_id_set]
                if not filtered_particles:
                    continue

                constituent_count = len(filtered_particles)

                # Extract properties into NumPy arrays in a single loop for efficiency.
                props = [
                    (p.eta(), p.phi(), p.e(), p.px(), p.py(), p.pz(), p.charge(), p.xProd(), p.yProd(), p.zProd())
                    for p in filtered_particles
                ]
                # Unzip the list of tuples into separate arrays.
                etas, phis, es, pxs, pys, pzs, charges, xvs, yvs, zvs = (np.array(prop) for prop in zip(*props))

                pts = np.sqrt(pxs**2 + pys**2)

                # Vectorised calculations.
                deltaRs = deltaR_vec(best_jet.eta(), best_jet.phi(), etas, phis)
                d0s = (xvs * pys - yvs * pxs) / np.maximum(pts, 1e-9)
                z0s = zvs - (xvs * pxs + yvs * pys) * (pzs / np.maximum(pts**2, 1e-9))

                d0_jet, z0_jet = np.sum(d0s), np.sum(z0s)
                d0_mean = d0_jet / constituent_count if constituent_count > 0 else 0.0
                z0_mean = z0_jet / constituent_count if constituent_count > 0 else 0.0
                deltaR_mean = np.mean(deltaRs)

                # Calculate the invariant mass of the jet.
                jet_mass_squared = np.sum(es)**2 - (np.sum(pxs)**2 + np.sum(pys)**2 + np.sum(pzs)**2)
                jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0
                q_jet = np.sum(charges)

                event_records.append((abs(h.id()), d0_mean, z0_mean, jet_mass, lxy, q_jet, deltaR_mean)) # Continue to find all in event

            elif consts:
                return constituents, h, best_jet # Return the first valid jet found
    
    except Exception as e:
        print(f'Error processing event: {e}')


def generate_events(pythia, jet_def, output_file, no_events, chunk_size, dtype):
    '''Main function to generate events and store them in an HDF5 file.'''
    start_time = time.time()
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

            new_records = single_event(pythia.event, jet_def)
            if new_records:
                buffer.extend(new_records)
                charm_events += len(new_records)

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
        p = subprocess.Popen(command) # Launch the worker process in the background.
        processes.append(p)
    
    print('All worker shards launched. Master process (shard 0) will now begin its work.')
    return processes


def merge_shards(output_file_base, num_shards, dtype, cleanup=True):
    '''Merge shard files into a single output file.'''
    print('\nMerging shard files into a single output file...')
    shard_files = []
    total_rows = 0
    # Determine the names of all shard files and calculate the total number of events.
    base, ext = os.path.splitext(output_file_base)
    for i in range(num_shards):
        shard_file = f'{base}_shard_{i}{ext}'
        if os.path.exists(shard_file):
            shard_files.append(shard_file)
            with h5py.File(shard_file, 'r') as f:
                total_rows += f['events'].shape[0]

    with h5py.File(output_file_base, 'w') as final_h5:
        # Create the final dataset with the correct total size.
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
    
    # Optional: Clean up temporary shard files after merging.
    if cleanup:
        print('Cleaning up temporary shard files...')
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
            except OSError as e:
                print(f'Error removing file {shard_file}: {e}')
        print('Cleanup complete.')