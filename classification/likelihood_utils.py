import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class FitResult:
    mu_hat: float
    sigma_up: float
    sigma_down: float
    mu_max: float
    mu_min: float
    mu_scan: np.ndarray
    dnll_scan: np.ndarray
    Z: float


@dataclass
class FitData:
    S: np.ndarray
    B_charm: np.ndarray
    B_other: np.ndarray
    B: np.ndarray
    D: np.ndarray
    bin_edges: np.ndarray


def load_fit_data(run_dir, scaling_dict=None, results_filename='test_results.npz', bins=50, inject_mu=1.0):
    """
    scaling_dict should look like:
    {
        'lumi_fb': 140,
        'charm': {'sigma_mb': 1.281e-2},
        'background': {'sigma_mb': 5.175}
    }
    """
    run_path = os.path.join('runs', run_dir)
    results_path = os.path.join(run_path, results_filename)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found.")
    
    data = np.load(results_path)
    y_true = data['y_true']
    probs = data['y_proba'][:, 2]

    mask_sig = (y_true == 2)
    mask_charm = (y_true == 1)
    mask_other = (y_true == 0)

    if scaling_dict is not None:
        lumi_mb = scaling_dict['lumi_fb'] * 1e9 # convert to mb
        
        n_charm_events = np.sum(mask_sig) + np.sum(mask_charm)
        n_bg_events = np.sum(mask_other)
        
        w_charm = (scaling_dict['charm']['sigma_mb'] * lumi_mb) / n_charm_events
        w_other = (scaling_dict['background']['sigma_mb'] * lumi_mb) / n_bg_events

        print(f"--- Scaling Check ---")
        print(f"Total events found -> Charm: {n_charm_events}, Background: {n_bg_events}")
        print(f"Weights -> Charm: {w_charm:.4f}, Other: {w_other:.4f}")
    else:
        w_charm = w_other = 1.0

    bin_edges = np.linspace(0, 1, bins + 1)

    S, _ = np.histogram(probs[mask_sig], bins=bin_edges, weights=np.full(np.sum(mask_sig), w_charm))
    B_charm, _ = np.histogram(probs[mask_charm], bins=bin_edges, weights=np.full(np.sum(mask_charm), w_charm))
    B_other, _ = np.histogram(probs[mask_other], bins=bin_edges, weights=np.full(np.sum(mask_other), w_other))
    
    B = B_other + B_charm
    
    expected = inject_mu * S + B
    D = np.random.poisson(expected).astype(float)
    
    print(f"Expected Yields -> S: {np.sum(S):.1f}, B: {np.sum(B):.1f}")
    
    return FitData(S, B_charm, B_other, B, D, bin_edges)


def poisson_nll(mu, S, B, D):
    mu_val = mu[0] if isinstance(mu, np.ndarray) and mu.size == 1 else mu
    expected = mu_val * S + B
    expected = np.maximum(expected, 1e-9)
    term = np.where(D > 0, D * np.log(D / expected), 0.0)
    return np.sum(term + expected - D)


def poisson_nll_jac(mu, S, B, D):
    mu_val = mu[0] if isinstance(mu, np.ndarray) and mu.size == 1 else mu
    expected = mu_val * S + B
    expected = np.maximum(expected, 1e-9)
    grad = np.sum(S * (1.0 - D / expected))
    return np.array([grad])


def perform_fit(S, B, D, mu_min=0.0, mu_max=2.0) -> FitResult:
    '''
    Finds best-fit mu_hat, asymmetric errors (up/down), and significance Z.
    '''
    res = minimize(poisson_nll, [1.0], args=(S, B, D), method='L-BFGS-B', jac=poisson_nll_jac, bounds=[(1e-5, None)], options={'ftol': 1e-12, 'gtol': 1e-8})
    mu_hat = res.x[0]
    min_nll = res.fun

    mu_scan = np.linspace(mu_min, mu_max, 100)
    dnll_scan = 2.0 * (np.array([poisson_nll(m, S, B, D) for m in mu_scan]) - min_nll)
    
    sigma_up, sigma_down = np.nan, np.nan
    from scipy.optimize import root_scalar

    def dnll_root(m):
        return 2.0 * (poisson_nll(m, S, B, D) - min_nll) - 1.0

    try:
        left_bound = 1e-5
        if dnll_root(left_bound) > 0:
            res_left = root_scalar(dnll_root, bracket=[left_bound, mu_hat], method='brentq')
            if res_left.converged:
                mu_min_1sig = res_left.root
                sigma_down = mu_hat - mu_min_1sig
                
        right_bound = mu_hat + 0.1
        while dnll_root(right_bound) <= 0 and right_bound < 100.0:
            right_bound += 1.0
            
        if dnll_root(right_bound) > 0:
            res_right = root_scalar(dnll_root, bracket=[mu_hat, right_bound], method='brentq')
            if res_right.converged:
                mu_max_1sig = res_right.root
                sigma_up = mu_max_1sig - mu_hat
    except Exception:
        pass

    nll_1 = poisson_nll(np.array([1.0]), S, B, D)
    q1 = 2.0 * (nll_1 - min_nll)
    if q1 < 0 and np.isclose(q1, 0, atol=1e-5):
        q1 = 0.0
        
    Z = np.sqrt(max(q1, 0.0))
        
    return FitResult(mu_hat, sigma_up, sigma_down, mu_max, mu_min, mu_scan, dnll_scan, Z)


def run_pull_study(S, B, inject_mu=1.0, n_toys=1000):
    mu_hats = []
    pulls = []

    expected = inject_mu * S + B
    
    for _ in range(n_toys):
        D_toy = np.random.poisson(expected).astype(float)
        res = perform_fit(S, B, D_toy)
    
        diff = res.mu_hat - inject_mu
        error = res.sigma_down if diff > 0 else res.sigma_up
        
        if error > 0:
            mu_hats.append(res.mu_hat)
            pulls.append(diff / error)
            
    return np.array(mu_hats), np.array(pulls)


def plot_pulls(mu_hats, pulls, run_dir, inject_mu=1.0): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(mu_hats, bins=50, color='blue', alpha=0.5)
    ax1.axvline(inject_mu, color='red', linestyle='--', label=rf'Truth ($\mu={inject_mu}$)')
    
    if len(mu_hats) > 0:
        mu_m, std_m = norm.fit(mu_hats)
        x_m = np.linspace(np.min(mu_hats), np.max(mu_hats), 100)
        p_m = norm.pdf(x_m, mu_m, std_m)
        
        # Scale PDF to match raw bin counts
        bin_width = (np.max(mu_hats) - np.min(mu_hats)) / 50.0
        scale_factor = len(mu_hats) * bin_width
        
        ax1.plot(x_m, p_m * scale_factor, 'k', linewidth=2, label=rf'Fit: $\mu={mu_m:.4f}, \sigma={std_m:.4f}$')
    ax1.set_title(r'Best-fit Signal Strength ($\hat{\mu}$)')
    ax1.set_xlabel(r'$\hat{\mu}$')
    ax1.legend()

    mu_p, std_p = norm.fit(pulls)
    x = np.linspace(-4, 4, 100)
    p = norm.pdf(x, mu_p, std_p)

    ax2.hist(pulls, bins=50, color='red', density=True, alpha=0.5)
    ax2.plot(x, p, 'k', linewidth=2, label=rf'Fit: $\mu={mu_p:.4f}, \sigma={std_p:.4f}$')
    ax2.set_xlim(-4, 4)
    ax2.set_title('Pull Distribution')
    ax2.set_xlabel(r'Pull $(\hat{\mu} - \mu_{inj}) / \sigma$')
    ax2.legend()
    
    plt.tight_layout()
    
    plots_dir = os.path.join(run_dir, 'pull_plots')
    os.makedirs(plots_dir, exist_ok=True)

    if inject_mu == 1.0:
        base_name = 'pull_study'
    else:
        base_name = f'pull_study_inject_{inject_mu}'
        
    ext = '.png'
    output_path = os.path.join(plots_dir, f'{base_name}{ext}')
    
    if os.path.exists(output_path):
        counter = 1
        while True:
            new_path = os.path.join(plots_dir, f'{base_name}_{counter}{ext}')
            if not os.path.exists(new_path):
                output_path = new_path
                break
            counter += 1
            
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")   


def plot_likelihood_scan(fit_result: FitResult, fit_data: FitData, run_dir: str, inject_mu: float = 1.0, y_max: float = 10.0):
    plots_dir = os.path.join(run_dir, 'likelihood_scans')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)

    samples = [fit_data.B_other, fit_data.B_charm, fit_data.S]
    colors = ['blue', 'green', 'red']
    class_labels = ['Background (Other)', 'Background (Charm)', 'Signal']

    bin_centers = 0.5 * (fit_data.bin_edges[:-1] + fit_data.bin_edges[1:])
    centers_list = [bin_centers, bin_centers, bin_centers]
    plt.hist(centers_list, bins=fit_data.bin_edges, weights=samples, stacked=True, color=colors, label=class_labels, alpha=0.5)
    plt.errorbar(bin_centers, fit_data.D, yerr=np.sqrt(fit_data.D), fmt='ko', label=rf'Mock Data (Injected $\mu={inject_mu}$)')
    
    best_fit_model = fit_result.mu_hat * fit_data.S + fit_data.B
    plt.step(fit_data.bin_edges, np.append(best_fit_model, best_fit_model[-1]), where='post', color='black', linestyle='--', label=rf'Best Fit ($\mu={fit_result.mu_hat:.2f}$)')
    plt.yscale('log')
    plt.xlabel(r'Neural Network Output $\mathbb{P}(\Lambda_c^+)$')
    plt.ylabel('Counts / Bin')
    plt.title(rf'Template Fit (Injected $\mu={inject_mu}$) - Run: {run_dir}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)

    plt.plot(fit_result.mu_scan, fit_result.dnll_scan, 'k.-', linewidth=2)
    plt.axvline(fit_result.mu_hat, color='r', linestyle='--', label=rf'$\hat{{\mu}} = {fit_result.mu_hat:.4f}$')
    plt.axhline(1.0, color='gray', linestyle=':', label=r'$\Delta(2NLL) = 1$ ($1\sigma$)')

    if not np.isnan(fit_result.sigma_down) and not np.isnan(fit_result.sigma_up): 
        plt.axvline(fit_result.mu_hat - fit_result.sigma_down, color='r', alpha=0.5, linestyle=':', label=rf'$-1\sigma$ ({fit_result.mu_hat - fit_result.sigma_down:.3f})')
        plt.axvline(fit_result.mu_hat + fit_result.sigma_up, color='r', alpha=0.5, linestyle=':', label=rf'$+1\sigma$ ({fit_result.mu_hat + fit_result.sigma_up:.3f})')
        
        diff = fit_result.mu_hat - inject_mu
        error = fit_result.sigma_down if diff > 0 else fit_result.sigma_up
        if error > 0:
            pull_val = diff / error
            plt.plot([], [], ' ', label=rf'Distance to $\mu_{{inj}}$: {pull_val:.2f}$\sigma$')
            
        plt.title(rf'Profile Likelihood Scan: $\hat{{\mu}} = {fit_result.mu_hat:.3f}_{{-{fit_result.sigma_down:.3f}}}^{{+{fit_result.sigma_up:.3f}}}$')
    else:
        plt.title(rf'Profile Likelihood Scan: $\hat{{\mu}} = {fit_result.mu_hat:.3f}$ ($1\sigma$ outside manual range)')
    
    plt.plot([], [], ' ', label=rf'Significance (from $\mu=1.0$): {fit_result.Z:.2f}$\sigma$')

    plt.xlabel(r'Signal Fraction $\mu$')
    plt.ylabel(r'$\Delta(2NLL)$')
    plt.ylim(-0.2, y_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if inject_mu == 1.0:
        base_name = 'likelihood_scan'
    else:
        base_name = f'likelihood_scan_inject_{inject_mu}'
        
    ext = '.png'
    output_path = os.path.join(plots_dir, f'{base_name}{ext}')
    
    if os.path.exists(output_path):
        counter = 1
        while True:
            new_path = os.path.join(plots_dir, f'{base_name}_{counter}{ext}')
            if not os.path.exists(new_path):
                output_path = new_path
                break
            counter += 1
            
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
