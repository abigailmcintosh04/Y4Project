import pythia8mc
import numpy as np
import math
import fastjet

# Vectorized deltaR between single (eta0,phi0) and arrays
def deltaR_vec(eta0, phi0, etas, phis):
    """Compute deltaR between (eta0,phi0) and arrays of etas, phis (NumPy arrays)."""
    dphi = np.abs(phis - phi0)
    # wrap
    mask = dphi > np.pi
    dphi[mask] = 2.0 * np.pi - dphi[mask]
    return np.sqrt((etas - eta0)**2 + dphi**2)

# Calculate distance in eta-phi space.
def deltaR(eta1, phi1, eta2, phi2):
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2*math.pi - dphi
    deta = eta1-eta2
    return math.sqrt(deta**2 + dphi**2)


pythia = pythia8mc.Pythia()

# Configure pp collisions at 13 TeV.
pythia.readString('Beams:idA = 2212')
pythia.readString('Beams:idB = 2212')
pythia.readString('Beams:eCM = 13000.')

# Enable charm quark production.
pythia.readString('HardQCD:gg2ccbar = on')
pythia.readString('HardQCD:qqbar2ccbar = on')

# Enable parton showering and hadronization.
pythia.readString("PartonLevel:ISR = on")
pythia.readString("PartonLevel:FSR = on")
pythia.readString("HadronLevel:Hadronize = on")

# Quiet Pythia output.
pythia.readString("Print:quiet = on")
pythia.readString("Next:numberShowEvent = 0")
pythia.readString("Next:numberShowInfo = 0")

# Use a random seed for the random number generator.
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 0") # 0 means use current time

# Initialize Pythia.
pythia.init()

# PDG IDs for charm hadrons and quarks.
hadron_id_set = {411, 421, 431, 4122, -411, -421, -431, -4122}  # Charm hadrons.
quark_id_set = {4, -4} # Charm quarks.

def get_c_quark_mother(particle, event, visited=None):
    """
    Recursively traverses up the mother list of a particle to find the
    first ancestor that is a charm quark. It prioritizes quarks from the
    hard process (status -22) but will return any c-quark ancestor if none
    is found from the hard process.
    A 'visited' set is used to prevent infinite loops in complex histories.
    """
    if visited is None:
        visited = set()

    # Prevent infinite recursion in complex event histories
    if particle.index() in visited:
        return None
    visited.add(particle.index())

    for mother_idx in particle.motherList():
        mother = event[mother_idx]
        # Base case: If the mother is a charm quark, we've found our target.
        if mother.id() in quark_id_set:
            return mother
        ancestor = get_c_quark_mother(mother, event, visited)
        if ancestor:
            return ancestor
    return None

jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

if pythia.next():
    fs_pseudojets = []
    hadrons = []
    for p in pythia.event:
        # Build list of final-state particles for jet clustering
        if p.isFinal():
            pj = fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_index(p.index())
            fs_pseudojets.append(pj)
        # Find charm hadrons
        if p.id() in hadron_id_set:
            hadrons.append(p)
    
    if not fs_pseudojets:
        print("No final state particles found in the event.")

    cs = fastjet.ClusterSequence(fs_pseudojets, jet_def)
    jets = cs.inclusive_jets(0.0)

    for h in hadrons:
        # Correctly find the original c-quark from the hard process
        c_quark = get_c_quark_mother(h, pythia.event)
        if not c_quark:
            print(f"Could not find hard-process c-quark ancestor for hadron {h.id()}.")
            continue

        # --- Efficient Jet Matching ---
        best_jet = None
        min_dr = 0.4 # The jet radius is a good initial cutoff
        for jet in jets:
            dr = deltaR(jet.eta(), jet.phi(), c_quark.eta(), c_quark.phi())
            if dr < min_dr:
                min_dr = dr
                best_jet = jet

        if not best_jet:
            print(f'No jet found within dR < 0.4 of the c-quark for hadron {h.id()}.')
            continue

        constituents = best_jet.constituents()
        if not constituents:
            continue

        print(f"\n--- Analysis for Hadron {h.name()} ({h.id()}) matched to Jet with pT={best_jet.pt():.2f} ---")
        for i, c in enumerate(constituents):
            p = pythia.event[c.user_index()]
            p_id = p.id()

            print(f'Particle index: {i}')
            print(f'Name: {p.name()} (ID: {p_id})')
            print(f'4-Momentum: ({p.px():.2f}, {p.py():.2f}, {p.pz():.2f}, {p.e():.2f}) GeV')
            print(f'pT: {p.pT():.2f} GeV, Mass: {p.m():.2f} GeV')
            print(f'Eta: {p.eta():.2f}, Phi: {p.phi():.2f}')





# # Generate one single event.
# if pythia.next():
#     print("--- Single Event Analysis ---")
#     # Loop through every particle in the event record.
#     for i, p in enumerate(pythia.event):
        # print(f"\n----- Particle Index: {i} -----")
        # print(f"  Name: {p.name()} (ID: {p.id()})")
        # print(f"  Status: {p.status()} {'(Final State)' if p.isFinal() else ''}")
        
        # # Kinematics
        # print("  Kinematics:")
        # print(f"    4-Momentum (px, py, pz, E): ({p.px():.2f}, {p.py():.2f}, {p.pz():.2f}, {p.e():.2f}) GeV")
        # print(f"    pT: {p.pT():.2f} GeV, Mass: {p.m():.2f} GeV")
        # print(f"    Rapidity (y): {p.y():.2f}, Pseudorapidity (eta): {p.eta():.2f}, Phi: {p.phi():.2f}")

        # # Charge and Color
        # print("  Charge & Color:")
        # print(f"    Charge: {p.chargeType()}/3 e, Is Charged: {p.isCharged()}")
        # print(f"    Color: {p.col()}, Anti-Color: {p.acol()}")

        # # Vertex and Lifetime
        # print("  Vertex & Lifetime:")
        # print(f"    Production Vertex (x, y, z, t): ({p.xProd():.2f}, {p.yProd():.2f}, {p.zProd():.2f}, {p.tProd():.2f}) mm")
        # print(f"    Decay Vertex (x, y, z, t): ({p.xDec():.2f}, {p.yDec():.2f}, {p.zDec():.2f}, {p.tDec():.2f}) mm")
        # print(f"    Proper Lifetime (tau): {p.tau():.4f} mm/c")

        # # Family Tree
        # print("  Family Tree (indices):")
        # print(f"    Mothers: {p.motherList()}")
        # print(f"    Daughters: {p.daughterList()}")