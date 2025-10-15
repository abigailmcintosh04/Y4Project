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
        ('id', 'i4'),
        ('state', 'S1'),
        ('pdg_id', 'i4'),
        ('px', 'f8'),
        ('py', 'f8'),
        ('pz', 'f8'),
        ('e', 'f8')
    ])
    dset = h5file.create_dataset(
        'particles', 
        shape=(0,),
        maxshape=(None,),
        dtype=dtypes,
        chunks=True 
    )
