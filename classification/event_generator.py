import pythia8mc
import h5py
import numpy as np
import math
import argparse
import fastjet

def deltaR(eta1, phi1, eta2, phi2):
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)

# Arguments for number of events and chunk size in command.
parser = argparse.ArgumentParser()
parser.add_argument('no_events', type=int)
parser.add_argument('chunk_size', type=int)
args = parser.parse_args()

no_events = args.no_events
chunk_size = args.chunk_size

output = 'collisions.h5'

# Jet definition.
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

# Initialize Pythia.
pythia = pythia8mc.Pythia()

# Physics stuff - pp collision at 13TeV.
pythia.readString('Beams:idA = 2212')
pythia.readString('Beams:idB = 2212')
pythia.readString('Beams:eCM = 13000.')

# Only charm production.
pythia.readString('HardQCD:gg2ccbar = on')
pythia.readString('HardQCD:qqbar2ccbar = on')

pythia.readString("PartonLevel:ISR = on")
pythia.readString("PartonLevel:FSR = on")
pythia.readString("HadronLevel:Hadronize = on")

# Quiet output.
pythia.readString("Print:quiet = on")
pythia.readString("Next:numberShowEvent = 0")
pythia.readString("Next:numberShowInfo = 0")

pythia.init()

hadron_id_set = {411, 421, 431, 4122, -411, -421, -431, -4122}  # Charm hadrons.
quark_id_set = {4, -4} # Charm quarks.

dtype = np.dtype([
    ('pdg_id_hadron', 'i4'),
    ('e_sum', 'f8'),
    ('pt_sum', 'f8'),
    ('d0_mean', 'f8'),
    ('jet_mass', 'f8'),
    ('lxy', 'f8'),
    ('q_jet', 'i4'),
])


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

    while charm_events < no_events:
        if not pythia.next():
            continue

        final_particles = []
        hadrons = []
        
        # Loop over the event to get hadrons and final particles.
        for p in pythia.event:
            if p.isFinal():
                final_particles.append(p)
            if p.id() in hadron_id_set:
                hadrons.append(p)
        
        # Build PseudoJets from the final_particles list.
        final_state_pseudojets = []
        for p in final_particles:
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_index(p.index())
            final_state_pseudojets.append(pj)
        
        # Cluster jets.
        cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
        jets = cluster_sequence.inclusive_jets(ptmin=0.0)

        for h in hadrons:
            mother_indices = h.motherList()
            quark_mothers = [pythia.event[i] for i in mother_indices if pythia.event[i].id() in quark_id_set]

            if not quark_mothers:
                continue

            c_quark = quark_mothers[0]

            best_jet = None
            min_dr = 0.4

            c_eta = c_quark.eta()
            c_phi = c_quark.phi()
            
            for jet in jets:
                dr = deltaR(jet.eta(), jet.phi(), c_eta, c_phi)
                if dr < min_dr:
                    min_dr = dr
                    best_jet = jet
            
            if best_jet is None:
                continue

            constituents = best_jet.constituents()
            if not constituents:
                continue

            e_jet = 0.0
            pt_jet = 0.0
            d0_jet = 0.0
            px_jet = 0.0
            py_jet = 0.0
            pz_jet = 0.0
            q_jet = 0

            lxy = math.sqrt(h.xDec()**2 + h.yDec()**2)

            constituent_count = 0

            for c in constituents:
                p = pythia.event[c.user_index()]
                p_id = p.id()

                if p_id in hadron_id_set or p_id in quark_id_set:
                    continue

                e_jet += p.e()
                pt_jet += p.pT()
                d0_jet += math.sqrt(p.xDec()**2 + p.yDec()**2)
                px_jet += p.px()
                py_jet += p.py()
                pz_jet += p.pz()
                q_jet += p.charge()
                
                constituent_count += 1
            
            if constituent_count > 0:
                d0_mean = d0_jet / constituent_count

                # Calculate invariant mass of jet.
                jet_mass_squared = e_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
                jet_mass = math.sqrt(jet_mass_squared) if jet_mass_squared > 0 else 0.0

                buffer.append((abs(h.id()), e_jet, pt_jet, d0_mean, jet_mass, lxy, q_jet))
                charm_events += 1

                # Write to file periodically.
                if charm_events % chunk_size == 0:
                    arr = np.array(buffer, dtype=dtype)
                    dset.resize(total_rows + len(arr), axis=0)
                    dset[total_rows:total_rows + len(arr)] = arr
                    total_rows += len(arr)
                    buffer = []

                if charm_events >= no_events:
                    break

    # Final flush of buffer.
    if buffer:
        arr = np.array(buffer, dtype=dtype)
        dset.resize(total_rows + len(arr), axis=0)
        dset[total_rows:total_rows + len(arr)] = arr
        total_rows += len(arr)

print('Events saved.')
