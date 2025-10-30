import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
args = parser.parse_args()
input_file = args.input_file

with open(os.path.join('runs', input, 'history.json'), 'r') as f:
    history = json.load(f)

val_data = np.load(os.path.join('runs', input, 'validation_results.npz'))
y_true = val_data['y_true']
y_pred = val_data['y_pred']
class_labels = val_data['class_data']

cm = np.load(os.path.join('runs', input, 'confusion_matrix.npy'))

# Training and validation loss.
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('runs', input, 'loss_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Training and validation accuracy.
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('runs', input, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation='vertical')
plt.title('Charm Hadron Classification — Confusion Matrix')
plt.savefig(os.path.join('runs', input, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
