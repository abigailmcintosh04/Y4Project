import fastjet 
import matplotlib.pyplot as plt
import os 
import numpy as np

from generator_utils import configure_pythia

pythia = configure_pythia()
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

if pythia.next():
    # Create a list of PseudoJets from final-state particles
    final_state_pseudojets = []
    for p in pythia.event:
        if p.isFinal():
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_index(p.index())
            final_state_pseudojets.append(pj)

    # Perform jet clustering
    cluster_sequence = fastjet.ClusterSequence(final_state_pseudojets, jet_def)
    jets = cluster_sequence.inclusive_jets(ptmin=0.0)

    point_list = []

    # Increase the figure size for better visibility
    fig, ax = plt.subplots(figsize=(10, 8))

    for jet in jets:
        print(f'\n--- Jet with pT={jet.pt():.2f}, eta={jet.eta():.2f}, phi={jet.phi():.2f} ---')
        constituents = jet.constituents()
        for c in constituents:
            p = pythia.event[c.user_index()]
            print(f'  Constituent {c.user_index()}: ID={p.id()}, pT={p.pT():.2f}, eta={p.eta():.2f}, phi={p.phi():.2f}')
            point_list.append((p.id(), p.eta(), p.phi()))

        # Make the jet circle thicker and more prominent
        circ = plt.Circle((jet.eta(), jet.phi_std()), 0.4, color='r', fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circ)
    
    # Group constituents by particle ID for color-coding
    points_by_id = {}
    for p_id, eta, phi in point_list:
        if p_id not in points_by_id:
            points_by_id[p_id] = []
        points_by_id[p_id].append((eta, phi))

    # Plot each particle type with a different color
    for particle_id, points in points_by_id.items():
        etas = [p[0] for p in points]
        phis = [p[1] for p in points]
        ax.scatter(etas, phis, label=f'ID {particle_id}', alpha=0.75, s=15) # s for marker size

    ax.set_aspect('equal', adjustable='box')

    # Dynamically set plot limits to zoom in on the jets
    if jets:
        all_constituent_etas = [pt[1] for pt in point_list]
        # Reduce the buffer to zoom in more tightly on the jets
        eta_min = min(all_constituent_etas) - 0.6
        eta_max = max(all_constituent_etas) + 0.6
        plt.xlim(eta_min, eta_max)

    plt.title('Jet Constituents in Eta-Phi Space')
    plt.xlabel('Eta')
    plt.ylabel('Phi')
    # Keep phi fixed as it's a circular coordinate
    plt.ylim(-np.pi, np.pi)
    plt.grid(True)
    plt.legend(title='Particle ID', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    # Use bbox_inches='tight' to prevent the legend from being cut off
    # Use a higher DPI for a higher resolution image
    plt.savefig(os.path.join('plots', 'jet_eta_phi.png'), bbox_inches='tight', dpi=300)

    