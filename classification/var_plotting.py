import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Argument for input file.
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
parser.add_argument('parameter', type=str)
parser.add_argument('x_min', type=float)
parser.add_argument('x_max', type=float)
args = parser.parse_args()
input_file = args.input_file
parameter = args.parameter
x_min = args.x_min
x_max = args.x_max

with h5py.File(input_file, 'r') as h5file:
    events = h5file['events'][:]

mask_1 = events['pdg_id_hadron'] == 411  # D+
mask_2 = events['pdg_id_hadron'] == 421  # D0
mask_3 = events['pdg_id_hadron'] == 431  # Ds+
mask_4 = events['pdg_id_hadron'] == 4122 # Lambdac+

plt.figure(figsize=(8,6))
plt.hist(events[parameter][mask_1], bins=np.linspace(x_min, x_max, 100), label='D+ (411)', color='blue', density=True, histtype='step')
plt.hist(events[parameter][mask_2], bins=np.linspace(x_min, x_max, 100), label='D0 (421)', color='orange', density=True, histtype='step')
plt.hist(events[parameter][mask_3], bins=np.linspace(x_min, x_max, 100), label='Ds+ (431)', color='green', density=True, histtype='step')
plt.hist(events[parameter][mask_4], bins=np.linspace(x_min, x_max, 100), label='Lambdac+ (4122)', color='red', density=True, histtype='step')
plt.xlabel(parameter)
plt.ylabel('Normalized Counts')
plt.title(f'Distribution of {parameter} for Different Charm Hadrons')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('plots', f'{parameter}_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()