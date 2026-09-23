"""Build a Cs-symmetric, NON-PLANAR water dimer starting guess (matching
the literature description of the "Smith stationary point 1" global
minimum: donor + H-bond axis in a mirror plane, acceptor O in that plane,
acceptor's two H's related by reflection through it -- i.e. out-of-plane,
not the simpler in-plane-splayed guess used elsewhere in this project),
then optimize it at wB97X/6-31G(d) with PySCF to get the actual DFT
stationary point at the paper's exact level of theory.

WHY A RECONSTRUCTION: Ranasinghe et al. 2025 (arXiv:2503.11537) cite "the
Smith stationary point 1" (ref. 20 in that paper, pointing to Gillan, Alfe
& Michaelides, "Perspective: How good is DFT for water?", J. Chem. Phys.
144, 130901 (2016) -- itself paywalled/inaccessible to this session) as
their starting geometry, optimized at wB97X/6-31G(d) with ORCA. No literal
deposited coordinates for this structure were found in the paper or its
SI (checked: the SI contains only training loss curves and quasi-harmonic
analysis tables, no structure files, no data/code availability statement
pointing to one). "Smith stationary point 1" itself traces to Smith,
Swanton, Pople, Schaefer & Radom, J. Chem. Phys. 92, 1240 (1990), which
characterized ten stationary points on the water dimer PES; SP1 is
independently confirmed by later literature (search results / secondary
sources, see RESULTS.md) to be the global minimum: a near-linear,
non-planar Cs-symmetric hydrogen-bonded structure. That qualitative
description (not literal coordinates) is what this script builds from,
then relaxes with a REAL wB97X/6-31G(d) optimization -- the DFT optimizer
should find the same true stationary point regardless of moderate
starting-guess imprecision, which is the entire point of using a real QM
engine here rather than hand-transcribing possibly-misremembered numbers.
This is a RECONSTRUCTION, not the paper's own structure -- documented as
such everywhere it's used downstream.

ENVIRONMENT: pyscf has no Windows wheels on PyPI (source-only, needs a C
compiler this machine doesn't have). Used a dedicated conda environment
instead (conda-forge does ship Windows builds):
    conda create -n mlip-audit-qm -c conda-forge python=3.11 pyscf ase numpy -y
    conda run -n mlip-audit-qm pip install pyberny   # geometry optimizer backend
Also: the conda-forge Windows pyscf build's direct-SCF integral buffer
allocation scales with OMP thread count and reliably failed with
malloc(3.2GB) on this 16-logical-core / ~4GB-available machine at the
default thread count, REGARDLESS of mol.max_memory. Fixed by capping
threads: run with `OMP_NUM_THREADS=2` in the environment. This is a
Windows-build-specific workaround, not something to carry over to a
Linux/mac pyscf install.
"""
import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.berny_solver import optimize

R_OH = 0.9584
ANGLE_HOH_DEG = 104.5
R_OO_GUESS = 2.98

half_angle = np.radians(ANGLE_HOH_DEG / 2.0)
theta = np.radians(ANGLE_HOH_DEG)

# Donor (monomer A): O at origin, bridging H exactly on +x (H-bond donor),
# free H at the H-O-H angle from it, both in the z=0 (mirror) plane.
o_a = np.array([0.0, 0.0, 0.0])
h_a_bridge = np.array([R_OH, 0.0, 0.0])
h_a_free = R_OH * np.array([np.cos(theta), np.sin(theta), 0.0])

# Acceptor (monomer B): O on the x-axis at R_OO_GUESS, IN the mirror
# plane. Its two H's point back toward the donor (bisector = -x) and are
# symmetric ABOVE/BELOW the mirror plane (+/-z), not splayed within it --
# this is the non-planar Cs symmetry the literature describes for the
# true global minimum, as opposed to a simpler planar guess.
o_b = np.array([R_OO_GUESS, 0.0, 0.0])
h_b1 = o_b + R_OH * np.array([-np.cos(half_angle), 0.0, np.sin(half_angle)])
h_b2 = o_b + R_OH * np.array([-np.cos(half_angle), 0.0, -np.sin(half_angle)])

atoms_guess = [o_a, h_a_bridge, h_a_free, o_b, h_b1, h_b2]
symbols = ["O", "H", "H", "O", "H", "H"]

atom_str = "; ".join(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for s, p in zip(symbols, atoms_guess))
print("Starting guess (Angstrom):")
print(atom_str)

mol = gto.M(atom=atom_str, basis="6-31g*", unit="Angstrom", max_memory=1500, verbose=4, charge=0, spin=0)
mf = dft.RKS(mol)
mf.xc = "wb97x"

mol_opt = optimize(mf, maxsteps=100)

print("\n\n=== OPTIMIZED GEOMETRY (Angstrom) ===")
coords = mol_opt.atom_coords(unit="Angstrom")
symbols_opt = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
for s, c in zip(symbols_opt, coords):
    print(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")

# Final single-point energy at the optimized geometry for the record.
mf_final = dft.RKS(mol_opt)
mf_final.xc = "wb97x"
e_final = mf_final.kernel()
print(f"\nFinal wB97X/6-31G(d) energy: {e_final} Hartree")

with open("geometries/smith_sp1_reconstructed.xyz", "w") as f:
    f.write(f"{mol_opt.natm}\n")
    f.write("Smith SP1 water dimer, RECONSTRUCTED (not from paper), optimized wB97X/6-31G(d) via PySCF+pyberny\n")
    for s, c in zip(symbols_opt, coords):
        f.write(f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n")
print("\nSaved: smith_sp1_reconstructed.xyz")
