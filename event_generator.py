import pythia8mc
import h5py
import numpy as np


no_events = 100000
chunk_size = 10000
output = 'collisions.h5'

pythia = pythia8mc.Pythia()

pythia.readString('Beams:idA = 2212')
pythia.readString('Beams:idB = 2212')
pythia.readString('Beams:eCM = 13000.')
# pythia.readString('HardQCD:all = on')
pythia.readString("PartonLevel:ISR = on")
pythia.readString("PartonLevel:FSR = on")
pythia.readString("HadronLevel:Hadronize = on")
pythia.readString("HardQCD:gg2ccbar = on")
pythia.readString("HardQCD:qqbar2ccbar = on")

pythia.readString("Next:numberShowInfo = 0")
pythia.readString("Print:quiet = on")
pythia.readString("Next:numberShowEvent = 0")
pythia.init()

hadron_ids = [411, 421, 431, 4122, -411, -421, -431, -4122]
quark_ids = [4, -4]

dtype = np.dtype([
        ('pdg_id_hadron', 'i4'),
        ('e_hadron', 'f8'),
        ('pdg_id_quark', 'i4'),
        ('e_quark', 'f8'),
])

with h5py.File(output, 'w') as h5file:
    dset = h5file.create_dataset(
        'particles', 
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=True 
    )

    event_id = 0
    total_rows = 0
    buffer = []

    while event_id < no_events:
        if not pythia.next():
            continue

        quarks = []
        hadrons = []

        for p in pythia.event:
            if p.id() in quark_ids:
                quarks.append(p)
            elif p.id() in hadron_ids:
                hadrons.append(p)

        charm_rows = []
        for c in quarks:
            for h in hadrons:
                if c.index() in h.motherList():
                    charm_rows.append((h.id(), h.e(), c.id(), c.e()))

        if charm_rows:
            buffer.extend(charm_rows)
            event_id += 1

        if event_id % chunk_size == 0 and buffer:
            arr = np.array(buffer, dtype=dtype)
            dset.resize(total_rows + len(arr), axis=0)
            dset[total_rows:total_rows + len(arr)] = arr
            total_rows += len(arr)
            buffer = []

    if buffer:
        arr = np.array(buffer, dtype=dtype)
        dset.resize(total_rows + len(arr), axis=0)
        dset[total_rows:total_rows + len(arr)] = arr
        total_rows += len(arr)

print('Events saved.')