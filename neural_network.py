import tensorflow as tf
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


input = 'collisions.h5'

with h5py.File(input, "r") as h5file:
    data = h5file["particles"][:]

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
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(
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

plt.figure(figsize=(8,8))
plt.scatter(y_true, y_pred, s=5, alpha=0.5)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()])
plt.xlabel('True hadron energy (GeV)')
plt.ylabel('Predicted hadron energy (GeV)')
plt.title('Predicted vs True Hadron Energies')
plt.grid(True)
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

