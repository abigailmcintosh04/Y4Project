import pythia8mc
import h5py
import numpy as np
import math
import argparse
import fastjet

def find_final_daughters(particle, event):
    """
    Recursively finds all final-state daughters of a given particle.
    """
    daughters = []
    
    # Get the index range of immediate daughters.
    d1_idx = particle.daughter1()
    d2_idx = particle.daughter2()

    # If no daughters, check if this particle is final.
    if d1_idx == 0:
        if particle.isFinal():
            daughters.append(particle)
        return daughters
    
    # Loop over all immediate daughters.
    for i in range(d1_idx, d2_idx + 1):
        daughter = event[i]
        # Recursively call this function for each daughter.
        daughters.extend(find_final_daughters(daughter, event))
    
    return daughters

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
DR_CUT_SQUARED = 0.4 ** 2

dtype = np.dtype([
    ('pdg_id_hadron', 'i4'),
    ('e_sum', 'f8'),
    ('pt_sum', 'f8'),
    ('d0_mean', 'f8'),
    ('m_reco', 'f8',)
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
            c_quark_pj = fastjet.PseudoJet(c_quark.px(), c_quark.py(), c_quark.pz(), c_quark.e())

            best_jet = None
            min_dr = 0.4
            
            for jet in jets:
                dr = jet.delta_R(c_quark_pj)
                if dr < min_dr:
                    min_dr = dr
                    best_jet = jet
            
            if best_jet is None:
                continue

            E_sum_reco = 0.0
            Px_sum_reco = 0.0
            Py_sum_reco = 0.0
            Pz_sum_reco = 0.0
            h_index = h.index()

            daughter_list = find_final_daughters(h, pythia.event)

            if daughter_list:
                for p_daughter in daughter_list:
                    E_sum_reco += p_daughter.e()
                    Px_sum_reco += p_daughter.px()
                    Py_sum_reco += p_daughter.py()
                    Pz_sum_reco += p_daughter.pz()

                m2_reco = E_sum_reco**2 - (Px_sum_reco**2 + Py_sum_reco**2 + Pz_sum_reco**2)
                m_reco = float(np.sqrt(m2_reco)) if m2_reco >= 0 else 0.0

            else:
                m_reco = 0.0

            constituents = best_jet.constituents()
            if not constituents:
                continue

            e_sum = 0.0
            pt_sum = 0.0
            d0_sum = 0.0
            constituent_count = 0

            for c in constituents:
                p = pythia.event[c.user_index()]
                p_id = p.id()

                if p_id in hadron_id_set or p_id in quark_id_set:
                    continue

                e_sum += p.e()
                pt_sum += p.pT()
                d0_sum += math.sqrt(p.xDec()**2 + p.yDec()**2)
                constituent_count += 1
            
            if constituent_count > 0:
                d0_mean = d0_sum / constituent_count
                buffer.append((abs(h.id()), e_sum, pt_sum, d0_mean, m_reco))
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
