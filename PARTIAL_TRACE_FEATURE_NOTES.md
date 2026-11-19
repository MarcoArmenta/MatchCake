# Partial trace feature: notes for future sessions

These notes describe features that live on the `dev` branch of the fork
[MarcoArmenta/MatchCake](https://github.com/MarcoArmenta/MatchCake) and are not in upstream MatchCake.
They are written for anyone, human or agent, who works on that branch or uses MatchCake with it checked
out. The README has a tutorial, and `docs/partial_trace_theory.md` the math.

## What was added

- `matchcake.operations.state_preparation.CovarianceStatePrepBase`: the interface of a state preparation
  that the device can start from. It provides `covariance_matrix`, `displacement` and `to_wires(wires)`.
  `ProductState` implements it (its `displacement` is new), and so does the new operation below. The
  device and its strategies test for this base class where they used to test for `ProductState`.
- `matchcake.operations.state_preparation.GaussianStatePrep(covariance_matrix, displacement=None,
  wires=None, validate=True)`: prepares a Gaussian state, pure or mixed, from its covariance matrix
  `(2n, 2n)` or `(B, 2n, 2n)`, with an optional displacement. It is a `StatePrepBase`, so it is the
  first operation of a circuit, inside or outside a QNode, and expectation values, probabilities,
  samples, shots, batches and gradients with respect to the covariance matrix come from the existing
  pipeline of the device. `to_wires` extends it to more wires in the same order, the others in |0>.
- `NonInteractingFermionicDevice.partial_trace(wires, **kwargs)` and
  `NonInteractingFermionicDevice.reduced_state(wires, *, reduced_wires=None, atol=None,
  fermionic=False)`. The first takes the wires to remove, the second the wires to keep. Both return a
  `MixedGaussianState` built from the principal submatrix of the device's extended covariance matrix
  on the Majorana indices of the kept wires (covariance and displacement). They work for any initial
  state that implements `CovarianceStatePrepBase`. The kept wires are relabeled `0..k-1`; the original
  labels are available as `state.source_wires`. A reduction that is not the qubit partial trace raises
  `ValueError` unless `fermionic=True`.
- `matchcake.states.MixedGaussianState` (exported as `matchcake.MixedGaussianState`): a container for a
  Gaussian state, possibly mixed and possibly displaced, held by its covariance matrix `(2m, 2m)` or
  `(B, 2m, 2m)` and its displacement. It does not run circuits. Members:
  - `state_prep(wires=None)`: the `GaussianStatePrep` that feeds the state into the next circuit.
  - `mode_polarizations`, `entropy(base=None)`, `purity`, `is_pure`, `density_matrix()` (Wick
    expansion, small systems only), `extended_covariance_matrix`, `displacement`, `is_displaced`,
    `from_polarizations(...)` (class method).
  - `reduced_state(wires, *, reduced_wires=None, fermionic=False)`, `partial_trace(wires, **kwargs)`:
    reductions of the state itself.
  - `insert(index, ancilla, *, wires=None)`, `append(ancilla, *, wires=None)`: add ancilla wires in a
    product state at a position of the wire list, like `list.insert` and `list.append`.
  - `sptm`, `mixed_modes`, `num_mixed_modes`, `num_pure_states`, `basis_states`, `weights`,
    `pure_states()`, `pure_state_operations(k)`, `ensemble_operations()`, `ensemble_wires`,
    `parity_wire`, `ensemble_observable(observable)`: the pure-state decomposition
    ρ = Σ_x w_x V_R|x⟩⟨x|V_R†, as a utility to analyze the state or to cross-check a result.
- `matchcake.utils.covariance` (`src/matchcake/utils/covariance.py`):
  `block_diagonalize_covariance(cov) -> (sptm, polarizations)` and
  `block_diagonal_covariance(polarizations)`. Backend and dtype preserving, batched, differentiable
  through a custom `torch.autograd.Function` whose backward maps gradients with respect to `(R, λ)`
  back to the covariance matrix and projects out gauge directions. Odd sizes (extended covariance
  matrices) are supported.

Minimal usage:

```python
dev = mc.NonInteractingFermionicDevice(wires=6)
dev.apply([qml.BasisState(np.array([1, 0, 1, 0, 1, 0]), wires=range(6)), *first_circuit_ops])
mixed = dev.partial_trace([4, 5])                      # MixedGaussianState on wires 0..3
print(mixed.entropy(), mixed.purity)

@qml.qnode(qml.device("nif.qubit", wires=4))
def follow_up():
    mixed.state_prep()                                 # GaussianStatePrep, first operation
    ...
    return qml.expval(observable)
```

## Design decisions

- **The state is prepared on the device, the container does not execute (redesign of 2026-11-18).**
  The first version of `MixedGaussianState` had its own `execute` with three modes, `expval`, `probs`,
  `sample` and `evolve`, and created devices internally. The upstream review of pull request #203
  (2026-11-05, changes requested) asked for a different structure: with that design a mixed state could
  not be used in a QNode, the class duplicated the device, and a dummy reference `ProductState` was
  needed to route the device strategies. The redesign follows the review: the device accepts any `CovarianceStatePrepBase`,
  `GaussianStatePrep` carries a covariance matrix (and a displacement) into the usual pipeline, and
  `MixedGaussianState` is reduced to the container described above. Removed: `execute`, `evolve`,
  `expval`, `probs`, `sample`, `device_kwargs` and their private helpers. The pure-state decomposition
  stays as a utility. Upstream history, for whoever prepares a pull request: #203 was merged into
  upstream `dev` on 2026-11-17 and reverted the same day (#245, "merged by mistake"), and #204 was
  closed. A new pull request should land the device-side change first (`CovarianceStatePrepBase`,
  `GaussianStatePrep`), then the container, then `partial_trace`, as the review suggests.
- **A reduction that is not the qubit partial trace must be requested (2026-11-18).** Also from the
  review. `reduced_state` and `partial_trace` raise `ValueError` when the kept wires are not a
  contiguous block, or when the state is displaced and the block does not start at the first wire.
  `fermionic=True` opts in. The fermionic partial trace has a precise meaning in terms of qubits: it is
  the qubit partial trace of the state whose traced-out wires were first moved behind the kept ones
  with fSWAP gates (verified numerically, with and without displacement).
- **Unphysical inputs are rejected (2026-11-18).** Also from the review. `from_polarizations` checks
  |λ| ≤ 1, and the normal-mode decomposition of a state raises `ValueError` when a polarization
  exceeds one (tolerance `PHYSICALITY_ATOL`). The check is in the decomposition, not in the
  constructor, so that building a state and preparing it on a device stays cheap.

- **Fermionic (mode) partial trace.** The reduced covariance matrix is the principal submatrix,
  so the Jordan-Wigner strings of the reduced system run through the kept wires only. This
  coincides with the qubit partial trace when the kept wires form a contiguous block (verified
  numerically against `qml.math.partial_trace`). For non-contiguous kept wires it differs by a
  `Z` on every traced-out wire between two `X`/`Y` factors of a Pauli word; the qubit version is
  not a Gaussian state and cannot be represented this way. The docs state this explicitly.
- **Product initial states, with a displacement when qubits are superposed (added 2026-11-18).**
  A product state with superposed qubits carries a displacement vector d_μ = ⟨c_μ⟩, the last
  column of the device's `extended_covariance_matrix`. Its reduced state is described by the
  principal submatrix of the extended covariance matrix on the kept Majorana indices and the
  parity index, that is by (Λ_A, d_A). `reduced_state` passes that displacement to
  `MixedGaussianState(covariance, displacement=...)`; it still raises `ValueError` for
  non-`ProductState` preparations. The earlier claim in these notes that the reduced state "is
  not Gaussian in the algebra of the kept wires" was too strong: it is a displaced Gaussian state,
  Gaussian for the 2m+1 operators (c_μ, parity), and everything is computed from the
  (2m+1) x (2m+1) extended matrix.
  - Qubit versus fermionic trace: an odd Majorana product keeps the Jordan-Wigner string from
    wire 0, so with a displacement the reduction equals the qubit partial trace only when the
    kept wires are a contiguous block starting at the first wire (trace out the last wires).
    Otherwise odd Pauli words are dressed with Z on the traced-out wires in front; even words and
    probabilities are unaffected. `MixedGaussianState.reduced_state` requires `fermionic=True` in
    that case. The exact qubit trace of a block that does not start at wire 0 is not a displaced
    Gaussian state of this kind and is not implemented.
  - Normal modes: `block_diagonalize_covariance` accepts odd sizes and returns R̃ ∈ SO(2m+1) with
    Λ̃ = R̃ᵀ(Λ_D ⊕ 0)R̃ (kernel vector as last row), with the custom backward extended to the
    mode/kernel border. Polarizations, entropy, purity and weights of a displaced state come
    from this decomposition, not from the 2m x 2m block.
  - Ensemble: the members of a displaced state live on m+1 wires, with a parity wire in front
    (`parity_wire`, `ensemble_wires`). There R̃ ⊕ 1 is an ordinary SPTM (`sptm`,
    `sptm_operation`), members are |0⟩|x⟩, even observables are measured as they are and odd
    ones with an X on the parity wire (`ensemble_observable`). Probabilities and samples in the
    computational basis only depend on Λ.
- **Adding ancilla wires: `insert` and `append` (added 2026-11-18).** The result is the qubit
  tensor product ρ ⊗ σ with the k ancilla wires at positions index..index+k-1. With F the wires of
  ρ in front of the ancilla B and G those behind it, the Majorana operators of the new register
  are c_μ (F), P_F b_ν (B) and c_μ P_B (G), so the new covariance and displacement are products of
  expectation values: Λ_FG is multiplied by ⟨P_B⟩, the F-B block is i⟨c_μ P_F⟩_ρ⟨b_ν⟩_σ, the B-G
  block is i⟨b_ν P_B⟩_σ⟨P_F c_μ⟩_ρ, and d' = (d_F, ⟨P_F⟩ d_σ, ⟨P_B⟩ d_G). The ρ factors are Pauli
  words (Z strings), read as Pfaffians of the extended covariance matrix
  (`_parity_string_correlators`); the σ factors are
  closed forms in the Bloch vectors.
  - Exactness, verified against dense tensor products at every position: an ancilla in a
    computational-basis state gives a Gaussian state anywhere. An ancilla with superposed qubits
    gives one exactly when the wires in front of it are in a pure state (index 0 always, any
    index of a product state, the end of a pure state, behind a pure block). Otherwise the
    tensor product violates Wick's theorem (one wire with Bloch vector r in front of an ancilla
    with Bloch vector s: ⟨C_0 C_1 C_2⟩ = i s_x, Wick gives i |r|² s_x) and `insert` raises
    `ValueError`. This is a property of the states, not of the implementation.
  - Consequence for repeated rounds with superposed ancillas: insertion wants the ancilla in
    front, a qubit-exact trace wants it behind. Moving it across with fSWAP gates is exact and is
    what the fermionic trace does anyway: the fermionic partial trace of any wires equals the
    qubit partial trace after the traced-out wires are moved behind the kept ones with fSWAPs
    (verified numerically, with and without displacement).
  - Upstream has no equivalent (checked 2026-11-18 on every upstream branch, issue and pull
    request): "ancilla" only appears there as an internal construction, in `ProductBasisSampling`
    and in the lifted covariance of the `swap_injection`, `magic` and `mc_compat_port` branches.
- **Gradients.** Values and gradients with respect to torch parameters of the first circuit match
  a dense reference to machine precision, through the reduced state, its `state_prep` and a second
  circuit. They flow linearly through the covariance matrix. The custom backward of the normal-mode
  decomposition is only used by the spectral quantities (entropy, purity, weights). It is exact at
  pure-mode degeneracies (λ = ±1, always present) because physical perturbations cannot push a
  polarization past 1, and not defined at an exact degeneracy between two *mixed* modes
  (measure-zero for random parameters).
- **Normal-mode ordering is arbitrary.** The decomposition orders modes by decreasing |λ| and can
  permute them relative to any "input" ordering; tests must not assume a specific order.
- **Wires of a reduced state must be consecutive integers** (matchgates are nearest-neighbour and
  the SPTM padding uses `make_wires_continuous`). Default `range(k)`.
- **Sampling of mixed states.** The default sampler of the device (`ProductBasisSampling`) only
  reads the extended covariance matrix, and its sequential measurement updates hold for mixed
  Gaussian states, so it samples a `GaussianStatePrep` input in the computational basis and in other
  single-qubit bases. This is tested against dense probabilities, as is the fallback sampler.
- Circular import avoided by importing `MixedGaussianState` lazily inside
  `NonInteractingFermionicDevice.reduced_state` (a `TYPE_CHECKING` import provides the
  annotation).

## Docs

- `docs/partial_trace_theory.md`: theory page (reduced state is Gaussian, normal modes and the
  pure-state decomposition, follow-up circuits on the device, gradients, product states with
  superposed qubits, adding ancilla wires, limitations). Written with the GitHub math conventions of
  `AGENTS.md` and registered in `sphinx/source/theory.rst`.
- Sphinx API pages: `sphinx/source/matchcake.states.rst`, and the new state-preparation modules in
  `sphinx/source/matchcake.operations.state_preparation.rst`.
- `README.md`: section "Mixed States, Partial Trace and Ancillas", a tutorial of the features that
  are only in this fork. Every code block was run and its outputs are the ones written in it.

## Tests

- `tests/test_utils/test_covariance.py` (`TestCovariance`): reconstruction, SO(2m) and SO(2m+1)
  property, block-diagonal form, degenerate inputs, backend/dtype preservation, `gradcheck` of the
  reconstruction and of a gauge-invariant loss, gradient at a pure degenerate point.
- `tests/test_operations/test_state_preparation/test_gaussian_state_prep.py` and
  `test_covariance_state_prep_base.py`: construction and validation of `GaussianStatePrep`, batches,
  backends, `to_wires` against dense states, rebinding of parameters, `ProductState.displacement`
  against brute force.
- `tests/test_devices/test_nif_device/test_nif_gaussian_state_prep.py`: a mixed state and a displaced
  mixed state, with covariance and displacement computed by brute force from a dense reduced density
  matrix, prepared on the device: QNode expectation values, full and marginal probabilities, both
  samplers, finite shots outside the computational basis, batches, part of the wires, `gradcheck`
  with respect to the covariance matrix and the displacement, QNode gradient.
- `tests/test_states/test_mixed_gaussian_state.py` (`TestMixedGaussianState`): construction and
  validation, decomposition properties, density matrix, entropy and purity vs dense, `state_prep`
  and circuits on it vs dense, chaining circuit and trace, opt-in fermionic reductions, unphysical
  inputs, the pure-state decomposition as a cross-check (with the parity wire for displaced states),
  `insert`/`append` vs dense tensor products, gradients vs a dense torch reference.
- `tests/test_devices/test_nif_device/test_nif_device_partial_trace.py`: contiguous traces vs
  `qml.math.partial_trace`, fermionic traces vs Jordan-Wigner-dressed expectations and vs the qubit
  trace after fSWAP gates, follow-up circuits, the reduced state fed to a QNode, batched devices,
  product-state inputs, error paths.

## Verification (2026-11-18)

| Check | Result |
|---|---|
| Full suite (`pytest tests`) | 10729 passed, 11 skipped |
| Coverage of `states/mixed_gaussian_state.py`, `utils/covariance.py`, `gaussian_state_prep.py`, `covariance_state_prep_base.py` | 100% |
| `ruff format --check src tests`, `ruff check src tests` | clean |
| `mypy` on the new and changed modules | clean |
| Every code block of the README tutorial | runs, outputs match |

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

- Exact qubit partial trace of a displaced state on a block of wires that does not start at the
  first wire. A block ending at the last wire should be a displaced Gaussian state for the
  reversed Jordan-Wigner order (parity wire behind instead of in front); a block in the middle
  needs something else.
- A qubit partial trace on non-contiguous wires is not Gaussian and would need a different
  representation.
- Second-order treatment of exact degeneracies between mixed modes in the decomposition
  backward, if the entropy or the weights are ever optimized at symmetric points.
- Syncing with upstream `dev`: upstream contains the commit of pull request #203 followed by its
  revert. This branch contains that commit as an ancestor, so a merge of upstream `dev` applies the
  revert and deletes or conflicts with the files of this feature. Resolve such a merge in favour of
  this branch for `src/matchcake/states`, `src/matchcake/utils/covariance.py` and their tests.
