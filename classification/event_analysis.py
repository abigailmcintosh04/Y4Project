import fastjet

from generator_utils import configure_pythia, single_event

pythia = configure_pythia()
jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

if pythia.next():
    constituents, h, best_jet = single_event(pythia.event, jet_def, consts=True)
    print(f"\n--- Analysis for Hadron {h.name()} ({h.id()}) matched to Jet with pT={best_jet.pt():.2f} ---")
    for i, c in enumerate(constituents):
        p = pythia.event[c.user_index()]
        p_id = p.id()

        print(f'Particle index: {i}')
        print(f'Name: {p.name()} (ID: {p_id})')
        print(f'4-Momentum: ({p.px():.2f}, {p.py():.2f}, {p.pz():.2f}, {p.e():.2f}) GeV')
        print(f'pT: {p.pT():.2f} GeV, Mass: {p.m():.2f} GeV')
        print(f'Eta: {p.eta():.2f}, Phi: {p.phi():.2f}')

