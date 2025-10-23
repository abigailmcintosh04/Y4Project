import tensorflow as tf
import h5py
import numpy as np


input = 'collisions.h5'

with h5py.File(input, "r") as h5file:
    data = h5file["particles"][:]

e_quark = data['e_quark']
e_hadron = data['e_hadron']
