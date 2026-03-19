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
lambda_c_ids = {4122, -4122}
other_charm_ids = {411, 421, 431, -411, -421, -431}  # Charm hadrons.
bottom_ids = {511, 521, 531, 5122, -511, -521, -531, -5122}  # Bottom hadrons.
all_heavy_ids = lambda_c_ids | other_charm_ids | bottom_ids


def calculate_d0(particle):
    '''
    Calculate the transverse impact parameter (d0) for a particle.
    '''
    pt = particle.pT()
    if pt < 1e-9:
        return 0.0
    return (particle.xProd() * particle.py() - particle.yProd() * particle.px()) / pt


def smear_d0(true_d0, pt_gev):
    '''
    Smears the true d0 to simulate ATLAS detector resolution.
    Uses the parameterization from ATL-PHYS-PUB-2015-051:
        sigma(d0) = a ⊕ b/pT (in mm, with pT in GeV)
    where a = 0.011 mm (11 um, intrinsic) and b = 0.073 mm*GeV (73 um*GeV, multiple scattering).
    '''
    a = 0.011
    b = 0.073
    sigma = np.sqrt(a**2 + (b / pt_gev)**2)
    return np.random.normal(true_d0, sigma)


def d0_significance(true_d0, pt_gev):
    '''
    Calculate the d0 significance for a particle.
    Uses the same resolution parameterization as smear_d0.
    '''
    a = 0.011
    b = 0.073
    sigma = np.sqrt(a**2 + (b / pt_gev)**2)
    d0_smeared = smear_d0(true_d0, pt_gev)
    significance = d0_smeared / sigma
    return significance, d0_smeared


def smear_pt(true_pt):
    '''
    Smears the true pT to simulate ATLAS Inner Detector resolution.
    Uses the parameterization from EPJC 74 (2014) 3130, Eq. 10:
        sigma(pT)/pT = r1 ⊕ r2 * pT
    where r1 = 0.017 (multiple scattering) and r2 = 0.000483 GeV^-1 (intrinsic).
    '''
    r1 = 0.017
    r2 = 0.000483
    sigma = true_pt * np.sqrt(r1**2 + (r2 * true_pt)**2)
    return np.random.normal(true_pt, sigma)


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


def configure_pythia(process='charm', seed=None):
    '''Configure and initialise the Pythia event generator.'''
    pythia = pythia8mc.Pythia()

    # Configure pp collisions at 13 TeV.
    pythia.readString('Beams:idA = 2212')
    pythia.readString('Beams:idB = 2212')
    pythia.readString('Beams:eCM = 13000.')

    if process == 'charm':
        # Charm production
        pythia.readString('HardQCD:hardccbar = on')
    elif process == 'background':
        pythia.readString('HardQCD:all = on')

    else:
        raise ValueError('Unknown process. Use "charm" or "background".')


    # Set a minimum pT for the hard process to ensure "jetty" events.
    pythia.readString('PhaseSpace:pTHatMin = 20.0')

    # Enable parton showering and hadronisation.
    pythia.readString('PartonLevel:ISR = on')
    pythia.readString('PartonLevel:FSR = on')
    pythia.readString('HadronLevel:Hadronize = on')

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


def jet_features(jet, event, class_label, lxy, d0_sig_cut=None):
    consts = jet.constituents()
    if not consts:
        return None
    
    valid_particles=[event[c.user_index()] for c in consts]

    px_jet, py_jet, pz_jet, e_jet, max_pt, charge_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    d0_values = []
    d0_sig_values = []

    for p in valid_particles:
        true_pt = p.pT()
        pt = smear_pt(true_pt)
        if pt < 1e-9:
            continue

        # Calculate true d0 and smear it to simulate detector resolution.
        true_d0 = calculate_d0(p)
        significance, smeared_d0 = d0_significance(true_d0, pt)

        # Optional d0 significance cut (applied after smearing).
        if (d0_sig_cut is not None) and (abs(significance) < d0_sig_cut):
            continue

        d0_values.append(smeared_d0)
        d0_sig_values.append(significance)
        charge_sum += int(p.charge())
        px_jet += p.px()
        py_jet += p.py()
        pz_jet += p.pz()
        e_jet += p.e()

        if pt > max_pt:
            max_pt = pt
    
    if len(d0_values) > 0:  
        d0_mean = np.mean(d0_values)
        d0_std = np.std(d0_values)
        d0_sig_mean = np.mean(np.abs(d0_sig_values))
        d0_sig_max = np.max(np.abs(d0_sig_values))

        # Calculate the invariant mass of the jet.
        jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
        jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

        # Calculate pT frac and jet pT (using filtered jet pT for consistency).
        filtered_jet_pt = math.sqrt(px_jet**2 + py_jet**2)
        pt_frac = max_pt / filtered_jet_pt if filtered_jet_pt > 0 else 0.0
        jet_pt = filtered_jet_pt

        n_tracks = len(d0_values)

        return (class_label, d0_mean, jet_mass, lxy, pt_frac, n_tracks,
                d0_sig_mean, d0_sig_max, jet_pt, d0_std, charge_sum)
    return None


def single_event(event, jet_def, ptmin, process='charm', d0_sig_cut=None):
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

        # Process hadrons.
        for p in event:
            p_id = p.id()
            if (process == 'charm') and (p_id in lambda_c_ids or p_id in other_charm_ids):
                daughters = p.daughterList()
                if daughters and all(event[d].id() not in all_heavy_ids for d in daughters):
                    hadrons.append(p)
            elif (process == 'background') and (p_id in bottom_ids):
                daughters = p.daughterList()
                if daughters and all(event[d].id() not in all_heavy_ids for d in daughters):
                    hadrons.append(p)

        cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
        jets = cluster_sequence.inclusive_jets(ptmin=ptmin)
        if not jets: return []

        if process == 'charm':
            for h in hadrons:
                min_dR = 0.3 
                best_jet = None
                for jet in jets:
                    dR = deltaR(h.eta(), h.phi(), jet.eta(), jet.phi())
                    if dR < min_dR:
                        min_dR = dR
                        best_jet = jet

                if best_jet:
                    lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)
                    class_label = 2 if abs(h.id()) == 4122 else 1
                    record = jet_features(best_jet, event, class_label, lxy, d0_sig_cut)
                    if record: event_records.append(record)

        elif process == 'background':
            for jet in jets:
                lxy = 0.0
                for h in hadrons:
                    if deltaR(h.eta(), h.phi(), jet.eta(), jet.phi()) < 0.3:
                        lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)
                        break 

                record = jet_features(jet, event, class_label=0, lxy=lxy, d0_sig_cut=d0_sig_cut)
                if record: event_records.append(record)
    
    except Exception as e:
        print(f'Error processing event: {e}')
    return event_records    
        

def generate_events(pythia, jet_def, output_file, no_events, chunk_size, dtype, ptmin, process='charm', d0_sig_cut=None):
    '''Main function to generate events and store them in an HDF5 file.'''
    start_time = time.time()
    last_report_time = start_time
    saved_events = 0

    with h5py.File(output_file, 'w') as h5file:
        # Pre-allocate the dataset; maxshape=None allows growing if needed.
        dset = h5file.create_dataset(
            'events',
            shape=(no_events,),
            maxshape=(None,),
            dtype=dtype,
            chunks=True
        )
        write_ptr = 0
        buffer = []

        # Main generation loop: continues until the desired number of charm events is found.
        while saved_events < no_events:
            if not pythia.next():
                continue

            new_records = single_event(pythia.event, jet_def, ptmin, process=process, d0_sig_cut=d0_sig_cut)
            if new_records:
                buffer.extend(new_records)
                saved_events += len(new_records)

            current_time = time.time()
            if current_time - last_report_time >= 30:
                elapsed_time = current_time - start_time
                print(f'Shard {os.getpid()}: Generated {saved_events} events in {elapsed_time:.2f} seconds.')
                last_report_time = current_time

            # Flush buffer to HDF5 file when it's full or at the end of generation.
            if len(buffer) >= chunk_size or (saved_events >= no_events and buffer):
                n_to_write = len(buffer)
                # Grow the dataset if needed to fit all events.
                if write_ptr + n_to_write > dset.shape[0]:
                    dset.resize((write_ptr + n_to_write,))

                arr = np.array(buffer[:n_to_write], dtype=dtype)
                dset[write_ptr : write_ptr + n_to_write] = arr
                write_ptr += n_to_write
                buffer = []
        
        # Write any final remaining events from the buffer.
        if buffer:
            n_to_write = len(buffer)
            if write_ptr + n_to_write > dset.shape[0]:
                dset.resize((write_ptr + n_to_write,))
            arr = np.array(buffer[:n_to_write], dtype=dtype)
            dset[write_ptr : write_ptr + n_to_write] = arr
            write_ptr += n_to_write

        # Resize dataset to exact number of events written.
        if write_ptr != dset.shape[0]:
            dset.resize((write_ptr,))

        # The final number of events is the number of rows written.
        saved_events = write_ptr
    
    duration = time.time() - start_time
    return saved_events, duration
                

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
        if hasattr(args, 'process') and args.process is not None:
            command.extend(['--process', args.process])
        if hasattr(args, 'd0_sig_cut') and args.d0_sig_cut is not None:
            command.extend(['--d0-sig-cut', str(args.d0_sig_cut)])
        if hasattr(args, 'temp_dir') and args.temp_dir is not None:
            command.extend(['--temp-dir', args.temp_dir])
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
                dset = f['events']
                n_rows = dset.shape[0]
                chunk_read_size = 500_000  # Process in 500k row chunks to save memory
                for i_start in range(0, n_rows, chunk_read_size):
                    i_end = min(n_rows, i_start + chunk_read_size)
                    data = dset[i_start:i_end]
                    rows_read = data.shape[0]
                    if rows_read > 0:
                        final_dset[write_ptr : write_ptr + rows_read] = data
                        write_ptr += rows_read

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