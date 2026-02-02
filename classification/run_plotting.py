import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Argument for input file.
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
args = parser.parse_args()
input_file = args.input_file

with open(os.path.join('runs', input_file, 'history.json'), 'r') as f:
    history = json.load(f)

val_data = np.load(os.path.join('runs', input_file, 'validation_results.npz'))
y_true = val_data['y_true']
y_pred = val_data['y_pred']
class_labels = val_data['class_labels']

cm = np.load(os.path.join('runs', input_file, 'confusion_matrix.npy'))

# Training and validation loss.
plt.figure(figsize=(8,5))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('runs', input_file, 'loss_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Training and validation accuracy.
plt.figure(figsize=(8,5))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('runs', input_file, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix.
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation='vertical', values_format='.2f')
plt.title('Charm Hadron Classification — Confusion Matrix (Normalised)')
plt.savefig(os.path.join('runs', input_file, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
