import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
import tensorflow as tf

# Mapping for nicer plot labels
pdg_map = {
    411: '$D^+$/$D^-$',
    421: '$D^0$/$\\bar{D}^0$',
    431: '$D_s^+$/$D_s^-$',
    4122: '$\\Lambda_c^+$/$\\bar{\\Lambda}_c$'
}

def get_label_name(pdg_id):
    return pdg_map.get(pdg_id, str(pdg_id))

def main():
    parser = argparse.ArgumentParser(description="Plot pairwise probability correlations for hadron classification.")
    parser.add_argument('run_name', type=str, help='Name of the run directory (e.g., 20260210-091303)')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for 2D histogram')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for the 2D histogram')
    args = parser.parse_args()

    run_dir = os.path.join('runs', args.run_name)
    data_path = os.path.join(run_dir, 'validation_results.npz')

    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    print(f"Loading data from {run_dir}...")
    data = np.load(data_path)
    y_true = data['y_true']
    class_labels = data['class_labels']

    # Check if probabilities are already saved; if not, try to generate them using the saved model.
    if 'y_proba' in data:
        y_proba = data['y_proba']
    else:
        print("y_proba not found in .npz file. Attempting to load model and generate probabilities...")
        model_path = os.path.join(run_dir, 'model.keras')
        if os.path.exists(model_path) and 'X_val' in data:
            try:
                model = tf.keras.models.load_model(model_path)
                X_val = data['X_val']
                y_proba = model.predict(X_val, verbose=0)
                print("Probabilities generated successfully.")
            except Exception as e:
                print(f"Error loading model or generating probabilities: {e}")
                return
        else:
            print("Error: Could not find model.keras or X_val to regenerate probabilities.")
            return

    # Create output directory for plots
    plots_dir = os.path.join(run_dir, 'prob_plots')
    os.makedirs(plots_dir, exist_ok=True)

    n_classes = len(class_labels)

    # Generate all pairs of class indices
    pairs = list(combinations(range(n_classes), 2))
    print(f"Generating {len(pairs)} pairwise probability plots...")

    for idx_x, idx_y in pairs:
        pdg_x = class_labels[idx_x]
        pdg_y = class_labels[idx_y]
        name_x = get_label_name(pdg_x)
        name_y = get_label_name(pdg_y)

        prob_x = y_proba[:, idx_x]
        prob_y = y_proba[:, idx_y]

        fig, ax = plt.subplots(figsize=(8, 7))

        h = ax.hist2d(prob_x, prob_y, bins=args.bins, range=[[0, 1], [0, 1]],
                       cmap=args.cmap, norm=matplotlib.colors.LogNorm(), cmin=1)
        plt.colorbar(h[3], ax=ax, label='Count (Log Scale)')

        ax.set_xlabel(f'P({name_x})')
        ax.set_ylabel(f'P({name_y})')
        ax.set_title(f'P({name_x}) vs P({name_y})')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Reference line where P(x) + P(y) = 1
        ax.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='P(x) + P(y) = 1')
        ax.legend(loc='upper right')

        output_file = os.path.join(plots_dir, f'prob_pair_{pdg_x}_vs_{pdg_y}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_file}")

    print("All pairwise plots generated.")

if __name__ == "__main__":
    main()
