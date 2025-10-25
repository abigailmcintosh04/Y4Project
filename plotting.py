import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('-l', '--loss', action='store_true')
parser.add_argument('-r', '--residuals', action='store_true')
parser.add_argument('-e', '--energies', action='store_true')
args = parser.parse_args()
input = args.input
loss = args.loss
residuals = args.residuals
energies = args.energies

with open(os.path.join('runs_energy', input, 'history.json'), 'r') as f:
    history = json.load(f)

val_data = np.load(os.path.join('runs', input, 'validation_results.npz'))
y_true = val_data['y_true']
y_pred = val_data['y_pred']

if energies:
    plt.figure(figsize=(8,8))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True hadron energy (GeV)')
    plt.ylabel('Predicted hadron energy (GeV)')
    plt.title('Predicted vs True Hadron Energies')
    plt.grid(True)
    plt.savefig('pred_energy.png', dpi=300, bbox_inches='tight')

if loss:
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

if residuals:
    calc_residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    plt.hist(calc_residuals, bins=100, range=(-20, 20))
    plt.xlabel('Residual (true - predicted)')
    plt.ylabel('Count')
    plt.title('Prediction Residuals')
    plt.savefig('residuals.png', dpi=300, bbox_inches='tight')

