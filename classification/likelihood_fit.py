import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

def poisson_nll(mu, S, B, D):
    # Expected number of events in each bin
    expected = mu * S + B
    # Avoid log(0) issues
    expected = np.maximum(expected, 1e-10)
    # NLL = -sum(D*ln(expected) - expected)
    nll = -np.sum(D * np.log(expected) - expected)
    return nll

def run_likelihood_fit(run_dir, results_path, mu_min, mu_max, inject_mu):
    run_path = os.path.join('runs', run_dir)
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    data = np.load(results_path)
    y_true = data['y_true']
    y_proba = data['y_proba']
    class_labels = data['class_labels']

    signal_pdg = 4122
    if signal_pdg not in class_labels:
        print(f"Error: Signal PDG {signal_pdg} not found in class labels.")
        return
        
    sig_idx = np.where(class_labels == signal_pdg)[0][0]
    
    probs = y_proba[:, sig_idx]
    
    is_signal = (y_true == sig_idx)
    sig_probs = probs[is_signal]
    bkg_probs = probs[~is_signal]

    n_bins = 50
    bins = np.linspace(0, 1.0, n_bins + 1)
    
    S, _ = np.histogram(sig_probs, bins=bins)
    B, _ = np.histogram(bkg_probs, bins=bins)
    
    # --- THE INJECTION HAPPENS HERE ---
    # We build the mock observation assuming the true signal is 'inject_mu' times larger
    D = (inject_mu * S) + B
    
    def objective(mu_val):
        return poisson_nll(mu_val[0], S, B, D)
        
    print(f"Performing Maximum Likelihood Fit (Injected true mu = {inject_mu})...")
    
    # Minimize the NLL (start the guess at 1.0, let the math find the injection!)
    initial_guess = [1.0]
    result = minimize(objective, initial_guess, bounds=[(0.0, None)])
    
    mu_hat = result.x[0]
    min_nll = result.fun
    
    print(f"Best fit signal strength (mu_hat) = {mu_hat:.4f}")
    
    nll_0 = poisson_nll(0.0, S, B, D)
    q0 = 2.0 * (nll_0 - min_nll)
    
    if q0 < 0 and np.isclose(q0, 0, atol=1e-5):
        q0 = 0.0
        
    Z = np.sqrt(max(q0, 0.0))
    print(f"Expected Significance (Wilks' Theorem) Z = {Z:.4f} sigma")
    
    plots_dir = os.path.join(run_path, 'prob_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    plt.hist(bins[:-1], bins, weights=B, histtype='stepfilled', alpha=0.3, color='blue', label='Background (B)')
    plt.hist(bins[:-1], bins, weights=S, bottom=B, histtype='stepfilled', alpha=0.3, color='red', label=r'Pythia Signal Theory (S, $\mu=1$)')
    plt.errorbar(bin_centers, D, yerr=np.sqrt(D), fmt='ko', label=f'Mock Data (Injected $\mu={inject_mu}$)')
    
    best_fit_model = mu_hat * S + B
    plt.step(bins, np.append(best_fit_model, best_fit_model[-1]), where='post', color='black', linestyle='--', label=rf'Best Fit ($\mu={mu_hat:.2f}$)')
    
    plt.yscale('log')
    plt.xlabel(r'Neural Network Output $\mathbb{P}(\Lambda_c^+)$')
    plt.ylabel('Counts / Bin')
    plt.title(f'Template Fit (Injected $\mu={inject_mu}$) - Run: {run_dir}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    
    mu_scan = np.linspace(mu_min, mu_max, 50)
    nll_scan = [poisson_nll(m, S, B, D) for m in mu_scan]
    
    dnll_scan = 2.0 * (np.array(nll_scan) - min_nll)
    
    plt.plot(mu_scan, dnll_scan, 'k.-', linewidth=2)
    plt.axvline(mu_hat, color='r', linestyle='--', label=rf'$\hat{{\mu}} = {mu_hat:.4f}$')
    plt.axhline(1.0, color='gray', linestyle=':', label=r'$\Delta(2NLL) = 1$ ($1\sigma$)')
    
    try:
        from scipy.interpolate import interp1d
        left_mask = mu_scan <= mu_hat
        right_mask = mu_scan >= mu_hat
        
        if np.any(dnll_scan[left_mask] >= 1.0):
            interp_left = interp1d(dnll_scan[left_mask], mu_scan[left_mask])
            mu_down = float(interp_left(1.0))
            plt.axvline(mu_down, color='gray', linestyle=':')
        else:
            mu_down = np.nan
            
        if np.any(dnll_scan[right_mask] >= 1.0):
            interp_right = interp1d(dnll_scan[right_mask], mu_scan[right_mask])
            mu_up = float(interp_right(1.0))
            plt.axvline(mu_up, color='gray', linestyle=':')
        else:
            mu_up = np.nan
            
        if not np.isnan(mu_down) and not np.isnan(mu_up):
            plt.title(rf'Profile Likelihood Scan: $\hat{{\mu}} = {mu_hat:.3f}_{{-{mu_hat - mu_down:.3f}}}^{{+{mu_up - mu_hat:.3f}}}$ | Significance: {Z:.2f}$\sigma$')
        else:
            plt.title(rf'Profile Likelihood Scan: $\hat{{\mu}} = {mu_hat:.3f}$ ($1\sigma$ outside manual range) | Significance: {Z:.2f}$\sigma$')
            
    except Exception as e:
        plt.title(rf'Profile Likelihood Scan | Significance: {Z:.2f}$\sigma$')

    plt.xlabel(r'Signal Strength $\mu$')
    plt.ylabel(r'$-2 \Delta \ln \mathcal{L}$')
    plt.ylim(0, 10)
    plt.xlim(mu_min, mu_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if inject_mu == 1.0:
        output_path = os.path.join(plots_dir, 'likelihood_scan.png')
    else:
        output_path = os.path.join(plots_dir, f'likelihood_scan_inject_{inject_mu}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform a binned maximum likelihood fit.")
    parser.add_argument('run_dir', type=str, help='Name of the run directory')
    parser.add_argument('--file', type=str, default='test_results.npz')
    parser.add_argument('--mu_min', type=float, default=0.96)
    parser.add_argument('--mu_max', type=float, default=1.04)
    parser.add_argument('--inject_mu', type=float, default=1.0)
    
    args = parser.parse_args()
    
    run_path = os.path.join('runs', args.run_dir)
    results_path = os.path.join(run_path, args.file)
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
    else:
        run_likelihood_fit(args.run_dir, results_path, args.mu_min, args.mu_max, args.inject_mu)