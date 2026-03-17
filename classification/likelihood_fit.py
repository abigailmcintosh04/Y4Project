import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

def poisson_nll(mu, S, B, D):
    """
    Computes the Poisson negative log-likelihood for a given signal strength mu.
    S: Signal template (array of bin counts)
    B: Background template (array of bin counts)
    D: Observed data (array of bin counts)
    """
    # Expected number of events in each bin
    expected = mu * S + B
    
    # Avoid log(0) issues
    expected = np.maximum(expected, 1e-10)
    
    # NLL = -sum(D*ln(expected) - expected - ln(D!))
    # We can omit the constant ln(D!) term since it doesn't affect the minimization and Delta_NLL
    nll = -np.sum(D * np.log(expected) - expected)
    return nll

def run_likelihood_fit(run_dir):
    run_path = os.path.join('runs', run_dir)
    results_path = os.path.join(run_path, 'validation_results.npz')
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    data = np.load(results_path)
    y_true = data['y_true']
    y_proba = data['y_proba']
    class_labels = data['class_labels']

    # Identify the Lambda_c+ (signal) class index
    signal_pdg = 4122
    if signal_pdg not in class_labels:
        print(f"Error: Signal PDG {signal_pdg} not found in class labels.")
        return
        
    sig_idx = np.where(class_labels == signal_pdg)[0][0]
    
    # Extract the probabilities for the signal class
    probs = y_proba[:, sig_idx]
    
    # Separate the events into Signal and Background true labels
    is_signal = (y_true == sig_idx)
    sig_probs = probs[is_signal]
    bkg_probs = probs[~is_signal]

    # Decide on binning
    n_bins = 50
    bins = np.linspace(0, 1.0, n_bins + 1)
    
    # Create histograms (templates)
    S, _ = np.histogram(sig_probs, bins=bins)
    B, _ = np.histogram(bkg_probs, bins=bins)
    
    # Create the Asimov dataset (Data = nominal Signal + Background)
    # This represents the case where our observation perfectly matches our expectation for mu=1
    D = S + B
    
    # If the total data is identically zero in some bins, it's fine (Poisson handles it)
    # but let's avoid issues by adding a tiny epsilon if needed, though D*log() handles zeros gracefully if D=0
    
    # Objective function to minimize (NLL as a function of mu)
    def objective(mu_val):
        return poisson_nll(mu_val[0], S, B, D)
        
    print("Performing Maximum Likelihood Fit...")
    
    # Minimize the NLL
    initial_guess = [1.0]
    result = minimize(objective, initial_guess, bounds=[(0.0, None)])
    
    # Extract best fit mu
    mu_hat = result.x[0]
    min_nll = result.fun
    
    print(f"Best fit signal strength (mu_hat) = {mu_hat:.4f}")
    
    # Evaluate NLL at mu = 0 (Background only hypothesis)
    nll_0 = poisson_nll(0.0, S, B, D)
    
    # Compute test statistic q0
    q0 = 2.0 * (nll_0 - min_nll)
    
    print(f"NLL(0) = {nll_0:.4f}")
    print(f"NLL(mu_hat) = {min_nll:.4f}")
    print(f"Test statistic q0 = {q0:.4f}")
    
    # Significance using Wilks' theorem (Z = sqrt(q0))
    # We suppress warnings if q0 < 0 due to precision issues when mu_hat ~ 0
    if q0 < 0 and np.isclose(q0, 0, atol=1e-5):
        q0 = 0.0
        
    Z = np.sqrt(max(q0, 0.0))
    print(f"Expected Significance (Wilks' Theorem) Z = {Z:.4f} sigma")
    
    # Plotting
    plots_dir = os.path.join(run_path, 'prob_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Templates and Data
    plt.subplot(2, 1, 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)
    
    # Plot Histograms as step functions
    plt.hist(bins[:-1], bins, weights=B, histtype='stepfilled', alpha=0.3, color='blue', label='Background (B)')
    plt.hist(bins[:-1], bins, weights=S, bottom=B, histtype='stepfilled', alpha=0.3, color='red', label=r'Signal (S, $\mu=1$)')
    
    # Plot Asimov Data
    plt.errorbar(bin_centers, D, yerr=np.sqrt(D), fmt='ko', label='Asimov Data (D = S+B)')
    
    # Best fit
    best_fit_model = mu_hat * S + B
    plt.step(bins, np.append(best_fit_model, best_fit_model[-1]), where='post', color='black', linestyle='--', label=rf'Best Fit ($\mu={mu_hat:.2f}$)')
    
    plt.yscale('log')
    plt.xlabel(r'Neural Network Output $\mathbb{P}(\Lambda_c^+)$')
    plt.ylabel('Counts / Bin')
    plt.title(f'Template Fit (Asimov Dataset) - Run: {run_dir}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: NLL Scan
    plt.subplot(2, 1, 2)
    mu_scan = np.linspace(max(0, mu_hat - 0.5), mu_hat + 0.5, 100)
    nll_scan = [poisson_nll(m, S, B, D) for m in mu_scan]
    
    # Convert NLL to delta NLL relative to minimum
    dnll_scan = 2.0 * (np.array(nll_scan) - min_nll)
    
    plt.plot(mu_scan, dnll_scan, 'k.-', linewidth=2)
    plt.axvline(mu_hat, color='r', linestyle='--', label=rf'$\hat{{\mu}} = {mu_hat:.4f}$')
    plt.axhline(1.0, color='gray', linestyle=':', label=r'$\Delta(2NLL) = 1$ ($1\sigma$)')
    
    # Find 1 sigma bounds
    try:
        from scipy.interpolate import interp1d
        # We split the curve into left of min and right of min
        left_mask = mu_scan <= mu_hat
        right_mask = mu_scan >= mu_hat
        
        if np.any(dnll_scan[left_mask] >= 1.0):
            interp_left = interp1d(dnll_scan[left_mask], mu_scan[left_mask])
            mu_down = interp_left(1.0)
            plt.axvline(mu_down, color='gray', linestyle=':')
        else:
            mu_down = np.nan
            
        if np.any(dnll_scan[right_mask] >= 1.0):
            interp_right = interp1d(dnll_scan[right_mask], mu_scan[right_mask])
            mu_up = interp_right(1.0)
            plt.axvline(mu_up, color='gray', linestyle=':')
        else:
            mu_up = np.nan
            
        if not np.isnan(mu_down) and not np.isnan(mu_up):
            plt.title(rf'Profile Likelihood Scan: $\hat{{\mu}} = {mu_hat:.3f}_{{-{mu_hat - mu_down:.3f}}}^{{+{mu_up - mu_hat:.3f}}}$ | Significance: {Z:.2f}$\sigma$')
        else:
            plt.title(rf'Profile Likelihood Scan: $\hat{{\mu}} = {mu_hat:.3f}$ | Significance: {Z:.2f}$\sigma$')
            
    except Exception as e:
        plt.title(rf'Profile Likelihood Scan | Significance: {Z:.2f}$\sigma$')

    plt.xlabel(r'Signal Strength $\mu$')
    plt.ylabel(r'$-2 \Delta \ln \mathcal{L}$')
    plt.ylim(0, max(5, np.max(dnll_scan)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'likelihood_scan.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform a binned maximum likelihood fit to extract signal strength and significance.")
    parser.add_argument('run_dir', type=str, help='Name of the run directory (e.g., 20240101-120000)')
    args = parser.parse_args()
    
    run_likelihood_fit(args.run_dir)
