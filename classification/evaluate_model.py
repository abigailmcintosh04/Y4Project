import tensorflow as tf
import h5py
import numpy as np
import joblib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=str, help="The folder containing your trained model")
parser.add_argument('test_file', type=str, help="The event file")
args = parser.parse_args()

run_dir = os.path.join('runs', args.run_dir)
full_file = os.path.join('collisions', args.test_file)

print(f"Loading data from {full_file}...")
with h5py.File(full_file, 'r') as f:
    data = f['events'][:]

X_test = np.vstack([data['d0_mean'], data['lxy'], data['jet_mass'], data['pt_frac'], data['n_tracks'],
              data['d0_sig_mean'], data['d0_sig_max'], data['jet_pt'], data['d0_std'],
              data['charge_sum']]).T.astype(np.float32)

y_raw = data['pdg_id_hadron']
y_binary = np.where(y_raw == 4122, 4122, 0)

class_labels = np.load(os.path.join(run_dir, "class_labels.npy"))
scaler = joblib.load(os.path.join(run_dir, "scaler.save"))

# Recreate integer mapping
y_true = np.where(y_binary == 4122, np.where(class_labels == 4122)[0][0], np.where(class_labels == 0)[0][0])

# Scale the unseen data
X_test_scaled = scaler.transform(X_test)

# Load the model and predict
print("Loading model and predicting...")
model = tf.keras.models.load_model(os.path.join(run_dir, "model.keras"))

y_proba = model.predict(X_test_scaled, batch_size=2048)
y_pred = np.argmax(y_proba, axis=1)

output_path = os.path.join(run_dir, "test_results.npz")
np.savez(output_path,
         y_true=y_true,
         y_pred=y_pred,
         y_proba=y_proba,
         class_labels=class_labels)

print(f"Saved predictions to {output_path}")