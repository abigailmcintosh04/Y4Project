import fastjet 
import matplotlib.pyplot as plt
import os 
import numpy as np
from datetime import datetime
from matplotlib.ticker import MultipleLocator

run_time = datetime.now().strftime("%Y%m%d-%H%M%S")

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
    jets = cluster_sequence.inclusive_jets(ptmin=10.0)

    point_list = []
    id_to_name_map = {}

    # Increase the figure size for better visibility
    fig, ax = plt.subplots(figsize=(10, 8))

    for jet in jets:
        print(f'\n--- Jet with pT={jet.pt():.2f}, eta={jet.eta():.2f}, phi={jet.phi():.2f} ---')
        constituents = jet.constituents()
        for c in constituents:
            p = pythia.event[c.user_index()]
            print(f'  Constituent {c.user_index()}: ID={p.id()}, pT={p.pT():.2f}, eta={p.eta():.2f}, phi={p.phi():.2f}')
            point_list.append((p.id(), p.eta(), p.phi(), p.pT()))
            if p.id() not in id_to_name_map:
                id_to_name_map[p.id()] = p.name()

        # Make the jet circle thicker and more prominent
        circ = plt.Circle((jet.eta(), jet.phi_std()), 0.4, color='r', fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circ)
    
    # Group constituents by particle ID for color-coding
    points_by_id = {}
    for p_id, eta, phi, pt in point_list:
        if p_id not in points_by_id:
            points_by_id[p_id] = []
        points_by_id[p_id].append((eta, phi, pt))

    # Plot each particle type with a different color
    for particle_id, points in points_by_id.items():
        etas = [p[0] for p in points]
        phis = [p[1] for p in points]
        pts = [p[2] for p in points]
        sizes = [30 * np.log(pt + 1) + 25 for pt in pts]
        particle_name = id_to_name_map.get(particle_id, str(particle_id))
        ax.scatter(etas, phis, label=f'{particle_name} ({particle_id})', alpha=0.75, s=sizes) # s for marker size

    # Highlight Charmed Hadrons
    charm_ids = {411, 421, 431, 4122}
    charm_particles = {}
    for p in pythia.event:
        if abs(p.id()) in charm_ids:
            if p.id() not in charm_particles:
                charm_particles[p.id()] = []
            charm_particles[p.id()].append(p)

    for pid, particles in charm_particles.items():
        etas = [p.eta() for p in particles]
        phis = [p.phi() for p in particles]
        pts = [p.pT() for p in particles]
        sizes = [30 * np.log(pt + 1) + 25 for pt in pts]
        name = particles[0].name()
        ax.scatter(etas, phis, s=sizes, edgecolors='black', linewidth=1, label=f'{name} (Charm)', zorder=10)

    ax.set_aspect('equal', adjustable='box')

    # Set fixed x-axis limits so the scale is consistent across different plots
    plt.xlim(-5, 5)

    plt.title('Jet Constituents in Eta-Phi Space', fontsize=20)
    plt.xlabel('Eta', fontsize=16)
    plt.ylabel('Phi', fontsize=16)
    # Keep phi fixed as it's a circular coordinate
    plt.ylim(-3.5, 3.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.legend(title='Particle', title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    # Use bbox_inches='tight' to prevent the legend from being cut off
    # Use a higher DPI for a higher resolution image
    plt.savefig(os.path.join('plots', f'jet_eta_phi_{run_time}.png'), bbox_inches='tight', dpi=300)

    