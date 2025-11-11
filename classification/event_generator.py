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

# Calculate distance in eta-phi space.
def deltaR(eta1, phi1, eta2, phi2):
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)

# Vectorized deltaR between single (eta0,phi0) and arrays
def deltaR_vec(eta0, phi0, etas, phis):
    """Compute deltaR between (eta0,phi0) and arrays of etas, phis (NumPy arrays)."""
    dphi = np.abs(phis - phi0)
    # wrap
    mask = dphi > np.pi
    dphi[mask] = 2.0 * np.pi - dphi[mask]
    return np.sqrt((etas - eta0)**2 + dphi**2)

# Command-line arguments for number of events and chunk size.
parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str, default='collisions.h5')
parser.add_argument('no_events', type=int)
parser.add_argument('chunk_size', type=int)
parser.add_argument('--shards', type=int, default=1, help='Total number of parallel shards to run.')
parser.add_argument('--shard-index', type=int, default=0, help='The index of this specific shard (0-based).')
args = parser.parse_args()

no_events_total = args.no_events
chunk_size = args.chunk_size

# --- Parallelization Logic ---
# If this is the master process (shard 0) and there are multiple shards,
# launch all the other worker processes in the background.
if args.shards > 1 and args.shard_index == 0:
    print(f"Master process launching {args.shards - 1} worker shards...")
    processes = []
    for i in range(1, args.shards):
        command = [
            sys.executable,  # The path to the current python interpreter
            __file__,        # The path to this script
            args.output_file,
            str(no_events_total),
            str(chunk_size),
            '--shards', str(args.shards),
            '--shard-index', str(i)
        ]
        # Launch the worker process. It will run in parallel.
        p = subprocess.Popen(command)
        processes.append(p)
    
    print("All worker shards launched. Master process (shard 0) will now begin its work.")

# Calculate how many events this specific shard is responsible for.
no_events = math.ceil(no_events_total / args.shards)

# --- Parallelization Logic for Output File ---
# Create a unique output filename for this shard.
if args.shards > 1:
    base, ext = os.path.splitext(args.output_file)
    output_file = f"{base}_shard_{args.shard_index}{ext}"
else:
    output_file = args.output_file


# Jet definition.
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# Initialize Pythia.
pythia = pythia8mc.Pythia()

# Configure pp collisions at 13 TeV.
pythia.readString('Beams:idA = 2212')
pythia.readString('Beams:idB = 2212')
pythia.readString('Beams:eCM = 13000.')

# Enable charm quark production.
pythia.readString('HardQCD:gg2ccbar = on')
pythia.readString('HardQCD:qqbar2ccbar = on')

# Enable parton showering and hadronization.
pythia.readString("PartonLevel:ISR = on")
pythia.readString("PartonLevel:FSR = on")
pythia.readString("HadronLevel:Hadronize = on")

# Quiet Pythia output.
pythia.readString("Print:quiet = on")
pythia.readString("Next:numberShowEvent = 0")
pythia.readString("Next:numberShowInfo = 0")

# Initialize Pythia.
pythia.init()

# PDG IDs for charm hadrons and quarks.
hadron_id_set = {411, 421, 431, 4122, -411, -421, -431, -4122}  # Charm hadrons.
quark_id_set = {4, -4} # Charm quarks.

# Data structure for the output HDF5 file.
dtype = np.dtype([
    ('pdg_id_hadron', 'i4'),
    ('e_sum', 'f8'),
    ('pt_sum', 'f8'),
    ('d0_mean', 'f8'),
    ('z0_mean', 'f8'),
    ('jet_mass', 'f8'),
    ('lxy', 'f8'),
    ('q_jet', 'i4'),
])

start_time = time.time()

# Open HDF5 file for writing.
with h5py.File(output_file, 'w') as h5file:
    # Pre-allocate the dataset to its full expected size to avoid slow resizing.
    dset = h5file.create_dataset(
        'events',
        shape=(no_events,),
        maxshape=(no_events,),
        dtype=dtype,
        chunks=True
    )

    write_ptr = 0 # Use a pointer to track our position in the dataset.
    buffer = []
    charm_events = 0

    # Main event generation loop. Continues until the desired number of charm events is reached.
    while charm_events < no_events:
        if not pythia.next():
            continue

        hadrons = []
        final_state_pseudojets = []
        
        # --- Combined Particle Loop ---
        # Create PseudoJets and find hadrons in a single pass.
        for p in pythia.event:
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
            if charm_events >= no_events:
                break

            mother_indices = h.motherList()
            quark_mothers = [pythia.event[i] for i in mother_indices if pythia.event[i].id() in quark_id_set]

            if not quark_mothers:
                continue

            c_quark = quark_mothers[0]

            # --- Vectorized Jet Matching ---
            # Find the jet closest to the charm quark efficiently.
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

            # (Calculations for jet properties remain the same as they were already fast)
            e_jet, pt_jet, d0_jet, z0_jet = 0.0, 0.0, 0.0, 0.0
            px_jet, py_jet, pz_jet, q_jet = 0.0, 0.0, 0.0, 0.0
            # Transverse decay length of the charm hadron.
            lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)
            constituent_count = 0

            # Loop over jet constituents to calculate jet properties.
            for c in constituents:
                p = pythia.event[c.user_index()]
                p_id = p.id()

                if p_id in hadron_id_set or p_id in quark_id_set:
                    continue

                e_jet += p.e()
                pt_jet += p.pT()
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
            
            if constituent_count > 0:
                d0_mean = d0_jet / constituent_count
                z0_mean = z0_jet / constituent_count

                jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
                jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

                buffer.append((abs(h.id()), e_jet, pt_jet, d0_mean, z0_mean, jet_mass, lxy, q_jet))
                charm_events += 1

        # --- Optimized Buffer Flushing ---
        # Flush buffer when it's full or at the very end.
        if len(buffer) >= chunk_size or (charm_events >= no_events and buffer):
            n_to_write = len(buffer)
            # Ensure we don't write past the pre-allocated space.
            if write_ptr + n_to_write > dset.shape[0]:
                n_to_write = dset.shape[0] - write_ptr
            
            arr = np.array(buffer[:n_to_write], dtype=dtype)
            dset[write_ptr : write_ptr + n_to_write] = arr
            write_ptr += n_to_write
            buffer = buffer[n_to_write:] # Keep any remainder.

    # Write any remaining events from the buffer to the file.
    if buffer:
        n_to_write = min(len(buffer), dset.shape[0] - write_ptr)
        arr = np.array(buffer, dtype=dtype)
        dset[write_ptr : write_ptr + n_to_write] = arr
        write_ptr += n_to_write

    # If we generated fewer events than expected, shrink the dataset to the actual size.
    if write_ptr < dset.shape[0]:
        dset.resize((write_ptr,))

end_time = time.time()
duration = end_time - start_time

print(f"Shard {args.shard_index}/{args.shards}: Event generation took {duration:.2f} seconds for {charm_events} events.")

# If this was the master process, wait for all worker processes to finish.
if args.shards > 1 and args.shard_index == 0:
    print("Master process finished its work. Waiting for worker shards to complete...")
    for p in processes:
        p.wait() # This will pause until the process 'p' has terminated.
    
    total_duration = time.time() - start_time
    print("All shards have completed successfully.")
    print(f"Overall time for all {args.shards} shards: {total_duration:.2f} seconds.")
