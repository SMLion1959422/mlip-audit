"""Model-loading harness: one factory that returns an ASE Calculator for
any of the three audited models, so test code never branches on model
identity.

    calc = get_calc("uma-s-1p1")
    atoms.calc = calc
    prepare_atoms_for_model(atoms, "uma-s-1p1")   # sets charge/spin if needed
    energy_eV = atoms.get_potential_energy()

Authentication: UMA's checkpoint is gated on HuggingFace. We never read or
store a token ourselves -- huggingface_hub's cached token (written by
`hf auth login`, normally at ~/.cache/huggingface/token) is picked up
automatically by fairchem/huggingface_hub. If it is missing, model loading
raises a clear error telling the user to run `hf auth login`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlip_audit.config import ESEN_CHECKPOINT_NAMES, MODEL_KEYS, MODEL_REQUIRES_CHARGE_SPIN

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def _default_device() -> str:
    """Return 'cuda' if a GPU is visible to torch, else 'cpu'."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_hf_login() -> None:
    """Raise a clear error if no cached HuggingFace token is found.

    UMA's checkpoint download requires having accepted the gated model's
    license and being logged in (`hf auth login`). We deliberately do not
    read the token value ourselves -- huggingface_hub does that internally.
    """
    from huggingface_hub import get_token

    if get_token() is None:
        raise RuntimeError(
            "No cached HuggingFace token found. UMA-S requires a HuggingFace "
            "login with access to the gated 'facebook/UMA' model. Run "
            "`hf auth login` (or `huggingface-cli login`) in this environment "
            "and accept the model license on huggingface.co, then retry."
        )


def get_calc(
    name: str, device: str | None = None, inference_settings: str | None = None
) -> "Calculator":
    """Build and return an ASE Calculator for the named model.

    Args:
        name: One of "ani2x", "mace-off23-small", "uma-s-1p1",
            "esen-conserving", "esen-direct".
        device: "cuda" or "cpu". Defaults to cuda if available, else cpu.
        inference_settings: fairchem-family models only (UMA, eSEN) --
            forwarded to `pretrained_mlip.get_predict_unit`. Defaults to
            "batch" (compile=False, merge_mole=False -- see _load_uma_s),
            the safe choice when a single calculator instance may see
            varying charge/spin/composition across calls (true of every
            Test 3 call in this project). Pass "turbo" ONLY when the
            caller guarantees this exact calculator instance will only
            ever be used on structures of ONE fixed composition (element
            counts, charge, and spin all constant for its whole lifetime)
            -- e.g. one MD trajectory. Turbo locks the predictor to the
            first system it sees; reusing a turbo calculator across a
            DIFFERENT composition silently produces wrong energies (this
            is why Test 3's multi-molecule/multi-charge usage must never
            pass "turbo" -- get a fresh calculator per fixed-composition
            system instead of sharing one). Ignored for ani2x/mace, which
            have no such setting. NOTE: "turbo"/"default" both require
            torch.compile's C++ backend (MSVC's cl.exe on Windows); this
            was never exercised on this project's local Windows/CPU
            machine (no compiler) -- only "batch" was validated locally.
            Verify turbo actually works in the target environment (e.g.
            Colab, which normally has gcc) before trusting it there.

    Returns:
        An ase.calculators.calculator.Calculator instance, ready to attach
        to an ase.Atoms object via `atoms.calc = calc`.

    Raises:
        ValueError: if `name` is not a recognized model key.
        RuntimeError: if a fairchem model is requested without a cached HF login.
    """
    if name not in MODEL_KEYS:
        raise ValueError(f"Unknown model {name!r}. Known models: {list(MODEL_KEYS)}")

    device = device or _default_device()

    if name == "ani2x":
        return _load_ani2x(device)
    elif name == "mace-off23-small":
        return _load_mace_off23_small(device)
    elif name == "uma-s-1p1":
        return _load_uma_s(device, inference_settings=inference_settings)
    elif name in ESEN_CHECKPOINT_NAMES:
        return _load_esen(name, device, inference_settings=inference_settings)

    raise AssertionError("unreachable")  # MODEL_KEYS check above is exhaustive


def _load_ani2x(device: str) -> "Calculator":
    import torch
    import torchani

    model = torchani.models.ANI2x(periodic_table_index=True)
    model = model.to(torch.device(device))
    return model.ase()


def _load_mace_off23_small(device: str) -> "Calculator":
    from mace.calculators import mace_off

    # float64 to match the LBFGS relaxation precision used elsewhere in this
    # codebase and avoid single-precision noise near the (very steep) short
    # O-O part of the scan.
    return mace_off(model="small", device=device, default_dtype="float64")


def _load_uma_s(device: str, inference_settings: str | None = None) -> "Calculator":
    return _load_fairchem_predictor("uma-s-1p1", device, inference_settings)


def _load_esen(name: str, device: str, inference_settings: str | None = None) -> "Calculator":
    checkpoint = ESEN_CHECKPOINT_NAMES[name]
    return _load_fairchem_predictor(checkpoint, device, inference_settings)


def _load_fairchem_predictor(
    checkpoint_name: str, device: str, inference_settings: str | None
) -> "Calculator":
    """Shared loader for every fairchem-family model (UMA, eSEN).

    IMPORTANT: by default (inference_settings=None -> "batch"), do NOT use
    "turbo" here. Turbo mode locks the predictor to the first system size
    it sees and silently produces wrong energies for any
    differently-composed structure evaluated later on the SAME predictor
    instance. Test 3 evaluates many differently-charged/differently-shaped
    structures per run and must never pass "turbo". Test 2/Test 4 (MD) DO
    pass "turbo" deliberately, per their own fixed-composition-per-
    trajectory contract -- see get_calc's docstring for the full contract.

    ALSO IMPORTANT (found running this project on Windows/CPU): the
    DEFAULT inference_settings ("default", and "turbo" too) both set
    compile=True, which requires torch.compile's C++ backend (MSVC's
    cl.exe on Windows) to JIT-compile inference kernels. On a machine
    without a C/C++ compiler toolchain, this raises
    torch._inductor.exc.InductorError: InvalidCxxCompiler at first
    inference, not at import/load time. "batch" is the one named preset
    with compile=False AND merge_mole=False (merge_mole assumes fixed
    composition/charge/spin across calls -- safe via auto-fallback when
    those change, per fairchem's own docs, but "batch" sidesteps that
    entirely). If running on a machine with a working C++ compiler (e.g.
    Colab normally has gcc), "turbo"/"default" should work and be faster;
    this was never exercised locally this session (no compiler here) --
    verify it actually works in the target environment before trusting it.
    """
    _check_hf_login()

    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    settings = inference_settings or "batch"
    predictor = pretrained_mlip.get_predict_unit(
        checkpoint_name, device=device, inference_settings=settings
    )
    return FAIRChemCalculator(predictor, task_name="omol")


def prepare_atoms_for_model(
    atoms: "Atoms", name: str, charge: int = 0, spin: int = 1
) -> None:
    """Attach charge/spin metadata to `atoms` if the named model needs it.

    UMA's `omol` task reads `atoms.info["charge"]` and `atoms.info["spin"]`
    (spin = multiplicity, i.e. 2S+1) on every call; without them the model
    silently falls back to defaults and produces wrong energies for charged
    or open-shell species. ANI-2x and MACE-OFF23 have no such input and
    this function is a no-op for them.

    Args:
        atoms: structure to annotate (mutated in place).
        name: model key, as passed to `get_calc`.
        charge: total charge of the structure.
        spin: spin multiplicity (2S+1); e.g. 1 for a closed-shell singlet.
    """
    if name not in MODEL_KEYS:
        raise ValueError(f"Unknown model {name!r}. Known models: {list(MODEL_KEYS)}")

    if MODEL_REQUIRES_CHARGE_SPIN[name]:
        atoms.info.update({"charge": charge, "spin": spin})
