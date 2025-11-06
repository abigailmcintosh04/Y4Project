import pythia8mc
import h5py
import numpy as np
import math
import argparse

# Arguments for number of events and chunk size in command.
parser = argparse.ArgumentParser()
parser.add_argument('no_events', type=int)
parser.add_argument('chunk_size', type=int)
args = parser.parse_args()

no_events = args.no_events
chunk_size = args.chunk_size

output = 'collisions.h5'

def deltaR(eta1, phi1, eta2, phi2):
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2 * math.pi - dphi
    return math.sqrt((eta1 - eta2)**2 + dphi**2)

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

        # --- Optimization 1: Single pass over the event record ---
        hadrons = []
        final_states = []
        for p in pythia.event:
            if p.id() in hadron_id_set:
                hadrons.append(p)
            if p.isFinal():
                final_states.append(p)
        
        # No need to build a separate 'quarks' list for the whole event

        for h in hadrons:
            mother_indices = h.motherList()
            if not mother_indices:
                continue

            # Find the c-quark mother
            c_quark = None
            for i in mother_indices:
                if pythia.event[i].id() in quark_id_set:
                    c_quark = pythia.event[i]
                    break # Found the first quark mother, stop looking
            
            if c_quark is None:
                continue

            c_eta = c_quark.eta()
            c_phi = c_quark.phi()

            # --- Optimization 2: Single-pass cone calculation ---
            e_sum = 0.0
            pt_sum = 0.0
            d0_sum = 0.0
            cone_count = 0

            for p in final_states:
                p_id = p.id() # Get ID once
                if p_id in hadron_id_set or p_id in quark_id_set:
                    continue

                # --- Optimization 3: Avoid sqrt in deltaR check ---
                deta = c_eta - p.eta()
                dphi = abs(c_phi - p.phi())
                if dphi > math.pi:
                    dphi = 2 * math.pi - dphi
                
                dr_sq = deta**2 + dphi**2

                if dr_sq <= DR_CUT_SQUARED:
                    e_sum += p.e()
                    pt_sum += p.pT()
                    d0_sum += math.sqrt(p.xDec()**2 + p.yDec()**2) # sqrt only for d0
                    cone_count += 1

            if cone_count > 0:
                d0_mean = d0_sum / cone_count
                buffer.append((abs(h.id()), e_sum, pt_sum, d0_mean))
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
