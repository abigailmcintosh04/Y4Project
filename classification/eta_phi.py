import fastjet 

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

    for jet in jets:
        print(f'\n--- Jet with pT={jet.pt():.2f}, eta={jet.eta():.2f}, phi={jet.phi():.2f} ---')
        constituents = jet.constituents()
        for c in constituents:
            p = pythia.event[c.user_index()]
            print(f'  Constituent {c.user_index()}: ID={p.id()}, pT={p.pT():.2f}, eta={p.eta():.2f}, phi={p.phi():.2f}')