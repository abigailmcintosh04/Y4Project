import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help="The original .h5 file")
parser.add_argument('run_dir', type=str, help="The folder containing validation_results.npz")
parser.add_argument('parameter', type=str, help="Variable to plot (e.g., d0_mean)")
parser.add_argument('x_min', type=float, help="Minimum x-axis value")
parser.add_argument('x_max', type=float, help="Maximum x-axis value")

# Custom Thresholds
parser.add_argument('--thresholds', nargs='*', type=str, default=[],
                    help="List of thresholds in format PDG=Threshold (e.g., 411=0.85 421=0.7)")
parser.add_argument('--default_threshold', type=float, default=0.5, 
                    help="Fallback threshold for particles not specified in --thresholds")

parser.add_argument('--log_scale', action='store_true')
args = parser.parse_args()

# --- SETUP ---
run_name = os.path.basename(os.path.normpath(args.run_dir))

custom_thresholds = {}
if args.thresholds:
    print(f"Applying custom thresholds for run {run_name}:")
    for item in args.thresholds:
        try:
            key_str, val_str = item.split('=')
            custom_thresholds[int(key_str)] = float(val_str)
            print(f"  -> Class {key_str}: {val_str}")
        except ValueError:
            print(f"  [WARNING] Skipping invalid format: '{item}'")

# --- LOAD DATA ---
results_path = os.path.join('runs', args.run_dir, "validation_results.npz")
if not os.path.exists(results_path):
    raise FileNotFoundError(f"Could not find {results_path}")

results = np.load(results_path)
y_proba = results['y_proba']
class_labels = results['class_labels']

full_file_path = os.path.join('collisions', args.input_file)
with h5py.File(full_file_path, 'r') as f:
    raw_data = f['events'][:]

y_raw = raw_data['pdg_id_hadron']
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# Re-split to match validation indices
_, raw_data_val, _, y_val_true = train_test_split(
    raw_data, y, test_size=0.2, stratify=y, random_state=42
)

# --- PLOTTING ---
var_dict = {
    'd0_mean': 'Mean Transverse Impact Parameter d0 (mm)',
    'lxy': 'Transverse Decay Length L_xy (mm)',
    'jet_mass': 'Jet Mass (GeV)',
    'pt_frac': 'Max Fraction of Track pT in Jet', 
}
label_name = var_dict.get(args.parameter, args.parameter)

plt.figure(figsize=(10, 7))

colors = {411: 'blue', 421: 'orange', 431: 'green', 4122: 'red'}
names = {411: 'D+', 421: 'D0', 431: 'Ds+', 4122: 'Lambdac+'}

bins = np.linspace(args.x_min, args.x_max, 100)

for i, pdg_id in enumerate(class_labels):
    
    cut_val = custom_thresholds.get(pdg_id, args.default_threshold)
    
    true_mask = (y_val_true == i)
    conf_mask = (y_proba[:, i] > cut_val)
    final_mask = true_mask & conf_mask
    
    data_to_plot = raw_data_val[args.parameter][final_mask]
    
    original_count = np.sum(true_mask)
    passed_count = len(data_to_plot)
    eff = (passed_count / original_count * 100) if original_count > 0 else 0
    
    label_text = f'{names.get(pdg_id, pdg_id)} (cut >{cut_val}, eff {eff:.1f}%)'
    
    plt.hist(
        data_to_plot, 
        bins=bins, 
        alpha=0.7, 
        density=True, 
        histtype='step', 
        linewidth=2,
        color=colors.get(pdg_id, 'black'),
        label=label_text,
        log=args.log_scale
    )

plt.xlabel(label_name)
plt.ylabel('Normalized Density')
plt.title(f'{label_name} Distribution (Run: {run_name})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(args.x_min, args.x_max)

# --- SAFE SAVING LOGIC ---
os.makedirs('plots', exist_ok=True)

# 1. Construct the base filename
base_name = f'optimized_cuts_{args.parameter}_{run_name}'
if args.log_scale:
    base_name += '_log_scale'

filename = f"{base_name}.png"
save_path = os.path.join('plots', filename)

# 2. Check if file exists and increment counter if needed
counter = 1
while os.path.exists(save_path):
    # If "plot.png" exists, try "plot_1.png", then "plot_2.png"...
    filename = f"{base_name}_{counter}.png"
    save_path = os.path.join('plots', filename)
    counter += 1

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")