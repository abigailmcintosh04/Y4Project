import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from likelihood_utils import perform_fit, plot_pulls, run_pull_study, load_fit_data
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform pull study.')
    parser.add_argument('run_dir', type=str)
    parser.add_argument('lumi_fb', type=float)
    parser.add_argument('sigma_s_mb', type=float)
    parser.add_argument('sigma_bg_mb', type=float)
    parser.add_argument('n_toys', type=int)
    parser.add_argument('--inject_mu', type=float, default=1.0)
    args = parser.parse_args()

    run_path = os.path.join('runs', args.run_dir)
    
    try:
        fit_data = load_fit_data(args.run_dir, args.lumi_fb, args.sigma_s_mb, args.sigma_bg_mb, inject_mu=args.inject_mu)
        mu_hats, pulls = run_pull_study(fit_data.S, fit_data.B, args.inject_mu, args.n_toys)
        plot_pulls(mu_hats, pulls, run_path, args.inject_mu)
    except Exception as e:
        print(f"Error running pull study: {e}")