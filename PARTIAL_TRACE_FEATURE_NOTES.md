# Partial trace feature: notes for future sessions

These notes describe a feature that lives on the branch `partial-trace-mixed-state` of the fork
[MarcoArmenta/MatchCake](https://github.com/MarcoArmenta/MatchCake) (branched from upstream `dev` on
2026-09-24, rebased onto upstream `dev` at `3cb8fc1` on 2026-10-23). It has not been reviewed or merged
upstream. They are written for anyone, human or agent, who picks up this branch or uses MatchCake with
it checked out.

## What was added

- `NonInteractingFermionicDevice.partial_trace(wires)` and
  `NonInteractingFermionicDevice.reduced_state(wires, *, reduced_wires=None, atol=None)`
  in `src/matchcake/devices/nif_device.py`. The first takes the wires to remove, the second the
  wires to keep. Both return a `MixedGaussianState` built from the principal submatrix of the
  device's covariance matrix on the Majorana indices of the kept wires. The kept wires are
  relabeled `0..k-1`; the original labels are available as `state.source_wires`. The returned
  state carries the device's `r_dtype`, `c_dtype` and `pfaffian_chunk_size` in `device_kwargs`.
- `matchcake.states.MixedGaussianState` (`src/matchcake/states/mixed_gaussian_state.py`,
  exported as `matchcake.MixedGaussianState`). A fermionic Gaussian state, possibly mixed,
  described by its covariance matrix `(2m, 2m)` or `(B, 2m, 2m)`. Main members:
  - `sptm`, `mode_polarizations`: the normal-mode decomposition Λ = Rᵀ Λ_D R with R ∈ SO(2m) and
    polarizations λ_j = ⟨Z_j⟩ of the normal modes.
  - `mixed_modes`, `num_mixed_modes`, `num_pure_states`, `basis_states`, `weights`,
    `pure_states()`, `pure_state_operations(k)`, `ensemble_operations()`: the pure-state
    decomposition ρ = Σ_x w_x V_R|x⟩⟨x|V_R†. Only mixed modes are enumerated, so there are at most
    2^(number of traced-out wires) members. Each member is a `ProductState` (basis state) followed
    by a `SingleParticleTransitionMatrixOperation` (R), which the device prepares natively.
  - `execute(operations, observable=None, output_type=None, *, mode="parallel", shots=None,
    device_kwargs=None, **kwargs)`: runs a follow-up circuit on the state and recombines.
    `mode="parallel"` prepares all members as one batched `ProductState` and executes once;
    `mode="sequential"` runs the members one at a time (this is the mode to use when the
    follow-up circuit has batched parameters); `mode="direct"` applies the circuit to the
    covariance matrix (Λ → R₂ᵀ Λ R₂) and reads the output from it with the same Pfaffian
    formulas, without enumerating the ensemble. Output types: `"expval"`, `"probs"`,
    `"samples"`, analytic or with `shots`.
  - `evolve(operations)`, `partial_trace(wires)`, `reduced_state(wires)`: covariance-level
    evolution and nested reductions (chaining trace → circuit → trace).
  - `expval(observable)`, `probs(wires=None)`, `sample(shots)`, `entropy(base=None)`,
    `purity`, `is_pure`, `density_matrix()` (Wick expansion, small systems only),
    `extended_covariance_matrix`, `from_polarizations(...)` (class method).
- `matchcake.utils.covariance` (`src/matchcake/utils/covariance.py`):
  `block_diagonalize_covariance(cov) -> (sptm, polarizations)` and
  `block_diagonal_covariance(polarizations)`. Backend and dtype preserving, batched,
  differentiable through a custom `torch.autograd.Function` whose backward maps gradients with
  respect to `(R, λ)` back to the covariance matrix and projects out gauge directions.

Minimal usage:

```python
dev = mc.NonInteractingFermionicDevice(wires=6)
dev.apply([qml.BasisState(np.array([1, 0, 1, 0, 1, 0]), wires=range(6)), *first_circuit_ops])
mixed = dev.partial_trace([4, 5])                      # MixedGaussianState on wires 0..3
expval = mixed.execute(follow_up_ops, observable, "expval", mode="parallel")
probs = mixed.execute(follow_up_ops, output_type="probs", mode="direct")
```

## Design decisions

- **Fermionic (mode) partial trace.** The reduced covariance matrix is the principal submatrix,
  so the Jordan-Wigner strings of the reduced system run through the kept wires only. This
  coincides with the qubit partial trace when the kept wires form a contiguous block (verified
  numerically against `qml.math.partial_trace`). For non-contiguous kept wires it differs by a
  `Z` on every traced-out wire between two `X`/`Y` factors of a Pauli word; the qubit version is
  not a Gaussian state and cannot be represented this way. The docs state this explicitly.
- **Computational-basis initial states only.** A product state with superposed qubits carries a
  displacement vector; its reduced state is Gaussian only in the extended Majorana algebra of the
  whole system, not in the algebra of the kept wires. `reduced_state` raises
  `NotImplementedError` for such states and `ValueError` for non-`ProductState` preparations.
- **Ensemble vs. direct evaluation.** The ensemble modes are what was asked for. The direct mode
  was added because the covariance matrix is linear in the state: it gives identical numbers,
  costs one execution instead of 2^r, and is the natural way to chain reductions.
- **Gradients.** Values and gradients of `execute` with respect to torch parameters of the first
  circuit match a dense `default.qubit` reference to machine precision in all three modes. The
  custom backward is exact at pure-mode degeneracies (λ = ±1, always present) because physical
  perturbations cannot push a polarization past 1. It is not defined at an exact degeneracy
  between two *mixed* modes (measure-zero for random parameters); `mode="direct"` has no such
  issue and is the recommended path for optimization.
- **Normal-mode ordering is arbitrary.** The decomposition orders modes by decreasing |λ| and can
  permute them relative to any "input" ordering; tests must not assume a specific order.
- **Wires of a reduced state must be consecutive integers** (matchgates are nearest-neighbour and
  the SPTM padding uses `make_wires_continuous`). Default `range(k)`.
- **Finite-shot expectation values.** With `shots`, `expval`/`probs` are estimated from
  computational-basis samples of the mixture, exactly as the device does. As on the device
  itself, a sample-based `expval` is only meaningful for observables diagonal in the
  computational basis: the device's own finite-shot estimate of `X(0) @ X(1)` gave 0.86 against
  an exact value of 0.0. This is a pre-existing limitation of the device, mirrored here and noted
  in the docstring, not fixed.
- Circular import avoided by importing `MixedGaussianState` lazily inside
  `NonInteractingFermionicDevice.reduced_state` (a `TYPE_CHECKING` import provides the
  annotation).

## Docs

- `docs/partial_trace_theory.md`: theory page (reduced state is Gaussian, normal modes and the
  pure-state decomposition, follow-up circuits, gradients, limitations). Written with the
  GitHub math conventions of `AGENTS.md` and registered in `sphinx/source/theory.rst`.
- Sphinx API pages: new `sphinx/source/matchcake.states.rst`, entries added to
  `sphinx/source/matchcake.rst` and `sphinx/source/matchcake.utils.rst`.
- `README.md`: new section "Tracing Out Wires and Continuing on the Reduced State" under
  "Quick Usage Preview" (its example was run and is genuinely mixed: entropy ≈ 0.28, 2 members).

## Tests

- `tests/test_utils/test_covariance.py` (`TestCovariance`): reconstruction, SO(2m) property,
  block-diagonal form, degenerate inputs (zero matrix, pure states, maximally mixed pairs),
  backend/dtype preservation, `gradcheck` of the reconstruction and of a gauge-invariant loss,
  gradient at a pure degenerate point along physical (rotation) perturbations.
- `tests/test_states/test_mixed_gaussian_state.py` (`TestMixedGaussianState`): construction and
  validation, decomposition properties, ensemble reconstructs the dense reduced density matrix,
  entropy/purity vs dense, all three `execute` modes vs dense for `expval` (Pauli sums,
  `BatchHamiltonian`, `Projector`), `probs` (full and marginal), `samples` (frequencies),
  finite-shot estimates, batched states and batched follow-up circuits, `evolve` and chaining,
  gradient vs dense torch reference in all three modes.
- `tests/test_devices/test_nif_device/test_nif_device_partial_trace.py`
  (`TestNonInteractingFermionicDevicePartialTrace`): contiguous traces vs
  `qml.math.partial_trace`, non-contiguous traces vs Jordan-Wigner-dressed expectations in the
  full system, follow-up circuit on non-contiguous wires vs the embedded free-fermion evolution,
  QNode-then-trace, batched devices, error paths (unknown wires, empty kept set, superposed
  product states, non-`ProductState` preparation).

## Verification (2026-09-24)

| Check | Result |
|---|---|
| Full suite (`pytest tests -n auto`) | 9899 passed, 1 skipped |
| Coverage of `states/mixed_gaussian_state.py` and `utils/covariance.py` | 100% |
| `ruff format --check src tests`, `ruff check src tests` | clean |
| `mypy` on the new modules | clean |
| Docs-related tests (`test_resolve_repository_links`, `test_check_math_rendering`, `test_sphinx_conf`) | pass |

## Environment notes (this macOS machine)

- The repository's `.venv` is broken: it points at a Python 3.12 framework that no longer exists.
  `uv` is not on `PATH`. Neither was modified.
- `torchpfaffian` has no macOS wheel, and its Rust extension fails to link against the macOS 27 SDK
  (`tapi error ... unknown architecture arm64e.x1-macos`). It builds against the older SDK:

  ```bash
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk \
    uv sync --locked --no-default-groups --group dev --python 3.12
  ```

  The verification above was run in a throwaway environment created that way (with
  `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` pointing outside the
  repository). The Rust kernel of `torch_pfaffian` was available in that environment; the package
  also has a pure-PyTorch fallback if the build is skipped.
- macOS is officially unsupported by MatchCake (see `README.md`), so CI (Linux, Windows) is the
  reference; nothing platform specific was added.

## Possible follow-ups (not done)

- Support superposed product initial states (needs the extended, displaced Gaussian formalism on
  the reduced system; research-level).
- A qubit partial trace on non-contiguous wires is not Gaussian and would need a different
  representation.
- Second-order treatment of exact degeneracies between mixed modes in the decomposition
  backward, if the ensemble modes are ever needed for optimization at symmetric points.
