import pythia8mc
import h5py
import numpy as np
import math
import argparse
import fastjet

# Calculate distance in eta-phi space.
def deltaR(eta1, phi1, eta2, phi2):
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)

# Command-line arguments for number of events and chunk size.
parser = argparse.ArgumentParser()
parser.add_argument('no_events', type=int)
parser.add_argument('chunk_size', type=int)
args = parser.parse_args()

no_events = args.no_events
chunk_size = args.chunk_size

# Output HDF5 file name.
output = 'collisions.h5'

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


# Open HDF5 file for writing.
with h5py.File(output, 'w') as h5file:
    dset = h5file.create_dataset(
        'events',
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=True
    )

    total_rows = 0
    buffer = []
    charm_events = 0
    total_rows = 0 # Keep track of the total number of rows written to the dataset.
    buffer = [] # A buffer to hold generated event data before writing to disk.
    charm_events = 0 # Counter for the number of charm hadron events generated.

    # Main event generation loop. Continues until the desired number of charm events is reached.
    while charm_events < no_events:
        if not pythia.next():
            # If Pythia fails to generate an event, skip to the next iteration.
            continue

        final_particles = []
        hadrons = []
        
        # Loop over the event to get hadrons and final particles.
        for p in pythia.event:
            if p.isFinal():
                final_particles.append(p)
            if p.id() in hadron_id_set:
                hadrons.append(p)
        
        # Build PseudoJets for jet clustering.
        final_state_pseudojets = []
        for p in final_particles:
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_index(p.index())
            final_state_pseudojets.append(pj)
        
        # Cluster jets.
        cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
        jets = cluster_sequence.inclusive_jets(ptmin=0.0)

        # Process each charm hadron in the event.
        for h in hadrons:
            mother_indices = h.motherList()
            quark_mothers = [pythia.event[i] for i in mother_indices if pythia.event[i].id() in quark_id_set]

            # Skip if the hadron's mother is not a charm quark.
            if not quark_mothers:
                continue

            c_quark = quark_mothers[0]

            # Find the jet closest to the charm quark.
            best_jet = None
            min_dr = 0.4 # Max deltaR for a jet to be matched to the quark.

            c_eta = c_quark.eta()
            c_phi = c_quark.phi()
            
            for jet in jets:
                dr = deltaR(jet.eta(), jet.phi(), c_eta, c_phi)
                if dr < min_dr:
                    min_dr = dr
                    best_jet = jet
            
            # Skip if no jet is matched.
            if best_jet is None:
                continue

            constituents = best_jet.constituents()
            if not constituents:
                continue

            e_jet = 0.0
            pt_jet = 0.0
            d0_jet = 0.0
            z0_jet = 0.0
            px_jet = 0.0
            py_jet = 0.0
            pz_jet = 0.0
            q_jet = 0
            # Transverse decay length of the charm hadron.
            lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)

            constituent_count = 0

            # Loop over jet constituents to calculate jet properties.
            for c in constituents:
                p = pythia.event[c.user_index()]
                p_id = p.id()

                # Exclude charm particles from jet property sums.
                if p_id in hadron_id_set or p_id in quark_id_set:
                    continue

                # Sum properties of jet constituents.
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
                    # Calculate transverse (d0) and longitudinal (z0) impact parameters.
                    d0 = (xv * py - yv * px) / pt
                    d0_jet += d0

                    z0 = zv - (xv * px + yv * py) * (pz / (pt**2))
                    z0_jet += z0
                
                constituent_count += 1
            
            if constituent_count > 0:
                # Calculate mean impact parameters.
                d0_mean = d0_jet / constituent_count
                z0_mean = z0_jet / constituent_count

                # Calculate jet's invariant mass from its 4-momentum.
                jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
                jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

                # Append features to buffer (use absolute PDG ID).
                buffer.append((abs(h.id()), e_jet, pt_jet, d0_mean, z0_mean, jet_mass, lxy, q_jet))
                charm_events += 1

                # Write buffer to file when chunk size is reached.
                if charm_events % chunk_size == 0:
                    arr = np.array(buffer, dtype=dtype)
                    dset.resize(total_rows + len(arr), axis=0)
                    dset[total_rows:total_rows + len(arr)] = arr
                    total_rows += len(arr)
                    buffer = []

                # Exit if desired number of events is reached.
                if charm_events >= no_events:
                    break

    # Write any remaining events from the buffer to the file.
    if buffer:
        arr = np.array(buffer, dtype=dtype)
        dset.resize(total_rows + len(arr), axis=0)
        dset[total_rows:total_rows + len(arr)] = arr
        total_rows += len(arr)

print('Events saved.')
