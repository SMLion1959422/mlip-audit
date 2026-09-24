"""Build Test 2's actual benchmark molecule: Ranasinghe et al. 2025's
349-atom artificial drug-like molecule (Fig. 2), a composite of structural
elements from clarithromycin, dexamethasone, diazepam, morphine,
penicillin, sildenafil, and tryptophan dipeptide.

UNLIKE the Smith SP1 water dimer (Test 3), this is NOT a reconstruction --
the paper deposits the EXACT SMILES string in its Supporting Information,
Appendix D ("SMILES of drug-like benchmark molecule"). Extracted here
programmatically from the fetched SI PDF text (both a `pdftotext -layout`
and a plain `pdftotext` extraction were cross-checked and agree character-
for-character) and validated: RDKit parses it to exactly 349 atoms
(180 heavy + 169 H) with formula C130H169ClFN15O31S2, EXACTLY matching the
paper's stated atom count (Fig. 2 caption: "the molecule with 349 atoms")
and its stated elements (H, C, N, O, S, F, Cl -- the paper notes ANI-1ccx
could not be used on this molecule specifically because of the S/F/Cl
atoms). An exact atom-count match by chance, if even one character of an
~500-character SMILES had been mis-transcribed across a PDF line-wrap,
would be very unlikely -- this is treated as strong (not certain)
evidence the string was extracted correctly, not proof.

3D embedding (RDKit ETKDG + MMFF94 optimization) is still ours, same as
every other molecule in this project -- the paper generated ITS starting
geometry "based on the SMILE strings ... using the GAFF2 force field,"
we use MMFF94 (RDKit's built-in) instead of GAFF2 (not available without
extra tooling); both are generic small-molecule force fields intended
only to produce a reasonable starting conformer, not a validated
prediction -- see mlip_audit/test2_md_stability.py, which LBFGS-optimizes
this starting geometry under each model's own PES before MD, exactly
matching the paper's own protocol ("before each ML simulation, the
geometry ... was optimized using the L-BFGS minimizer").
"""

import time

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Extracted from the paper's SI (Appendix D), cross-checked between a
# `pdftotext -layout` and a plain `pdftotext` extraction of the same PDF
# (identical result from both). See this module's docstring for the
# atom-count/formula validation that supports this being correct.
BENCHMARK_SMILES = (
    "C([C@@H]1C[C@H]2[C@@H]3CCC4=C(C(=O)C=C[C@@]4([C@]3([C@H](C[C@@]2([C@]"
    "1(C(=O)CO)O)C)O)F)CCCc1nn(c2c1nc([nH]c2=O)c1c(ccc(c1)S(=O)(=O)N1CCN(CC1)C"
    "CNC(=O)[C@@H](NC(=O)C)Cc1c[nH]c2c1cccc2)O[C@@]1(C[C@H](O[C@@H]2[C@H](C("
    "=O)O[C@H](CC)[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]2C)O[C"
    "@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)(C)OC)C)C)O)(C)O)C)O[C@H]([C@@H]1"
    "O)C)C)CCN1[C@H]2[C@@H]3C=C[C@H](O)[C@H]4[C@@]3(c3c(O4)c(ccc3C2)O)CC1)CC"
    "(=O)N[C@H]1[C@@H]2N([C@H](C(C)(S2)C)C(=O)O)C1=O)CC1=NCC(=O)N(C)c2c1cc"
    "(cc2)Cl"
)

EXPECTED_TOTAL_ATOMS = 349
EXPECTED_FORMULA = "C130H169ClFN15O31S2"
EXPECTED_ELEMENTS = {"C", "H", "N", "O", "S", "F", "Cl"}

EMBED_SEED = 0
OUT_PATH = "geometries/drug_like_benchmark_349atoms.xyz"


def main():
    mol = Chem.MolFromSmiles(BENCHMARK_SMILES)
    if mol is None:
        raise RuntimeError("RDKit failed to parse the benchmark SMILES -- do not proceed.")

    formula = rdMolDescriptors.CalcMolFormula(mol)
    print(f"Parsed OK. Formula (heavy-atom SMILES, RDKit default H handling): {formula}")

    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    print(f"Total atoms (with explicit H): {n_atoms} (expected {EXPECTED_TOTAL_ATOMS})")
    print(f"Elements: {sorted(elements)} (expected {sorted(EXPECTED_ELEMENTS)})")

    if n_atoms != EXPECTED_TOTAL_ATOMS:
        raise RuntimeError(
            f"Atom count mismatch: got {n_atoms}, paper states {EXPECTED_TOTAL_ATOMS}. "
            "Do NOT proceed -- the extracted SMILES is probably wrong (a mis-transcribed "
            "character from the PDF line-wrap). Re-extract and re-check before building anything."
        )
    if elements != EXPECTED_ELEMENTS:
        raise RuntimeError(
            f"Element set mismatch: got {sorted(elements)}, expected {sorted(EXPECTED_ELEMENTS)}. "
            "Do NOT proceed."
        )
    if formula != EXPECTED_FORMULA:
        raise RuntimeError(
            f"Formula mismatch: got {formula}, expected {EXPECTED_FORMULA}. Do NOT proceed."
        )
    print("All validation checks passed (atom count, elements, formula).")

    print(f"Embedding {n_atoms} atoms (ETKDG, seed={EMBED_SEED})...")
    t0 = time.time()
    embed_result = AllChem.EmbedMolecule(mol, randomSeed=EMBED_SEED, useRandomCoords=True)
    if embed_result != 0:
        raise RuntimeError("RDKit 3D embedding failed -- do not proceed with a fallback silently.")
    print(f"Embedded in {time.time() - t0:.1f}s")

    print("MMFF94 force-field optimization (starting-geometry cleanup only -- "
          "LBFGS under each model's own PES happens later, per protocol)...")
    t0 = time.time()
    ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    print(f"MMFF optimize result={ff_result} (0=converged) in {time.time() - t0:.1f}s")

    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    with open(OUT_PATH, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(
            "Ranasinghe et al. 2025 349-atom drug-like benchmark molecule "
            "(arXiv:2503.11537 SI Appendix D SMILES), RDKit ETKDG+MMFF94 "
            f"3D embedding, seed={EMBED_SEED}\n"
        )
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            f.write(f"{symbols[i]} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
