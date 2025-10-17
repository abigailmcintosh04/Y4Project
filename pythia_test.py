import pythia8mc
import h5py
import numpy as np


no_events = 10000
chunk_size = 100
output = 'collisions.h5'

pythia = pythia8mc.Pythia()
pythia.readString('Beams:idA = 2212')
pythia.readString('Beams:idB = 2212')
pythia.readString('Beams:eCM = 13600')
pythia.readString('HardQCD:all = on')
pythia.init()

with h5py.File(output, 'w') as h5file:
    dtypes = np.dtype([
        ('id', 'i8'),
        ('pdg_id_hadron', 'i4'),
        ('pdg_id_quark'),
        ('px', 'f8'),
        ('py', 'f8'),
        ('pz', 'f8'),
        ('e_hadron', 'f8'),
        ('e_quark', 'f8'),
    ])
    dset = h5file.create_dataset(
        'particles', 
        shape=(0,),
        maxshape=(None,),
        dtype=dtypes,
        chunks=True 
    )

hadron_ids = [411, 421, 431, 4122, -411, -421, -431, -4122]

event_id = 0
total_rows = 0

while event_id < no_events:
    rows = []

    for _ in range(min(chunk_size, no_events - event_id)):
        if not pythia.next():
            continue

        quarks = []
        hadrons = []

        for p in pythia.event:
            if abs(p.id()) == 4:
                p.append(quarks)
            elif p.id() in hadron_ids:
                p.append(hadrons)

        for q in quarks:
            for h in hadrons:
                mothers = h.motherList()
                if q.index() in mothers:
                    rows.append((event_id, h.id(), q.id(), h.px(), h.py(), h.pz(), h.e(), q.e()))

    event_id += 1
