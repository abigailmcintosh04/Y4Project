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

hadron_ids = [411, 421, 431, 4122, -411, -421, -431, -4122]  # Charm hadrons.
quark_ids = [4, -4] # Charm quarks.

dtype = np.dtype([
    ('pdg_id_hadron', 'i4'),
    ('e_sum', 'f8'),
    ('pt_sum', 'f8'),
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

        hadrons = [p for p in pythia.event if p.id() in hadron_ids]
        quarks = [p for p in pythia.event if p.id() in quark_ids]
        final_states = [p for p in pythia.event if p.isFinal()]

        for h in hadrons:
            mother_indices = h.motherList()
            quark_mothers = [pythia.event[i] for i in mother_indices if pythia.event[i].id() in quark_ids]

            if not quark_mothers:
                continue

            c_quark = quark_mothers[0]
            c_eta = c_quark.eta()
            c_phi = c_quark.phi()

            # Particles in deltaR <= 0.4 cone from the c quark.
            cone_particles = [
                p for p in final_states
                if p.id() not in hadron_ids and p.id() not in quark_ids
                and deltaR(c_eta, c_phi, p.eta(), p.phi()) <= 0.4
            ]

            if not cone_particles:
                continue

            e_sum = sum(p.e() for p in cone_particles)
            pt_sum = sum(p.pT() for p in cone_particles)

            buffer.append((abs(h.id()), e_sum, pt_sum))
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
