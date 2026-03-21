import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
from likelihood_utils import plot_likelihood_scan, perform_fit, load_fit_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform a binned maximum likelihood fit.")
    parser.add_argument('run_dir', type=str, help='Name of the run directory')
    parser.add_argument('--results_path', type=str, default='test_results.npz')
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--mu_min', type=float, default=0.9)
    parser.add_argument('--mu_max', type=float, default=1.1)
    parser.add_argument('--y_max', type=float, default=10)
    parser.add_argument('--inject_mu', type=float, default=1.0)
    parser.add_argument('--bg_weight', type=float, default=404.855, help="Weight applied to non-charm background events")
    
    args = parser.parse_args()
    
    run_path = os.path.join('runs', args.run_dir)
    results_path = os.path.join(run_path, args.results_path)
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
    else:
        scaling_dict = {
            'lumi_fb': 1,
            'charm': {'sigma_mb': 1.281e-2},
            'background': {'sigma_mb': 5.175}
        }
        
        fit_data = load_fit_data(args.run_dir, scaling_dict=scaling_dict, results_filename=args.results_path, bins=args.bins, inject_mu=args.inject_mu)
        fit_result = perform_fit(fit_data.S, fit_data.B, fit_data.D, args.mu_min, args.mu_max) 
        plot_likelihood_scan(fit_result, fit_data, run_path, args.inject_mu, args.y_max)