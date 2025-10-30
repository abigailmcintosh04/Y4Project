import tensorflow as tf
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import json
import argparse

# Arguments for number of events and chunk size in command.
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
args = parser.parse_args()
input_file = args.input_file

# input_file = 'collisions_cone.h5'

with h5py.File(input_file, 'r') as f:
    data = f['events'][:]

run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("runs", run_time)
os.makedirs(run_dir, exist_ok=True)

X = np.vstack([data['e_sum'], data['pt_sum']]).T.astype(np.float32)
y_raw = data['pdg_id_hadron']

# Encode hadron PDG IDs as integer classes.
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
class_labels = encoder.classes_
n_classes = len(class_labels)

# Split between train and validation.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardise.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=512,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

cm = confusion_matrix(y_val, y_pred)

with open(os.path.join(run_dir, "history.json"), "w") as f:
    json.dump(history.history, f)

np.savez(os.path.join(run_dir, "validation_results.npz"),
         y_true=y_val,
         y_pred=y_pred,
         X_val=X_val,
         class_labels=class_labels)

np.save(os.path.join(run_dir, "confusion_matrix.npy"), cm)

model.save(os.path.join(run_dir, "model.keras"))
np.save(os.path.join(run_dir, "class_labels.npy"), class_labels)

metadata = {
    "val_loss": float(val_loss),
    "val_accuracy": float(val_acc),
    "n_classes": int(n_classes),
    "class_labels": [int(c) for c in class_labels],
    "timestamp": run_time,
    "input_file": input_file
}
with open(os.path.join(run_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Training complete.")
