import tensorflow as tf
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import argparse

# Arguments for number of events and chunk size in command.
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()
input = args.input

# input = 'collisions.h5'

with h5py.File(input, "r") as h5file:
    data = h5file["particles"][:]

run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("runs_energies", run_time)
os.makedirs(run_dir, exist_ok=True)

e_quark = data['e_quark']
e_hadron = data['e_hadron']

x = e_quark.astype(np.float32).reshape(-1, 1)
y = e_hadron.astype(np.float32).reshape(-1, 1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y)

x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_scaled, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
])

optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimiser, loss='mse', metrics=['mae'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=1024,
    verbose=1,
)

val_loss, val_mae = model.evaluate(x_val, y_val)
print(f"Validation MAE: {val_mae:.4f}")

y_pred_scaled = model.predict(x_val)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_val)

with open(os.path.join(run_dir, "history.json"), "w") as f:
    json.dump(history.history, f)

np.savez(os.path.join(run_dir, "validation_results.npz"),
         y_true=y_true, y_pred=y_pred, x_val=x_val)




