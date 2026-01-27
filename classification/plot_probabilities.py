import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Mapping for nicer plot labels
pdg_map = {
    411: 'D+/D-',
    421: 'D0/D0_bar',
    431: 'Ds+/Ds-',
    4122: 'Lambdac+/Lambdac_bar'
}

def get_label_name(pdg_id):
    return pdg_map.get(pdg_id, str(pdg_id))

def main():
    parser = argparse.ArgumentParser(description="Plot probability histograms for hadron classification.")
    parser.add_argument('run_name', type=str, help='Name of the run directory (e.g., 20240101-120000)')
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

    # Iterate over each class to treat it as the "True" class
    for i, true_pdg in enumerate(class_labels):
        true_name = get_label_name(true_pdg)
        
        # Select events where the true label is class i
        mask = (y_true == i)
        events_probs = y_proba[mask]
        
        if len(events_probs) == 0:
            print(f"No validation events found for True Class {true_name} ({true_pdg})")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot the probability distributions for this set of events being identified as class j
        for j, pred_pdg in enumerate(class_labels):
            pred_name = get_label_name(pred_pdg)
            
            # Extract the probability column for class j
            probs_j = events_probs[:, j]
            
            # Determine line style: solid for the correct class, dashed for others
            linestyle = '-' if i == j else '--'
            linewidth = 2 if i == j else 1.5
            
            plt.hist(probs_j, bins=50, range=(0, 1), density=True, histtype='step', 
                     linewidth=linewidth, linestyle=linestyle, label=f'P({pred_name})')
            
        plt.title(f'Network Confidence when True Particle is {true_name}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Normalised Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(plots_dir, f'probs_true_{true_pdg}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_file}")

    print("All plots generated.")

if __name__ == "__main__":
    main()
