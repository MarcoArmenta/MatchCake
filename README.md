# MatchCake

<p align="center"><img src="https://github.com/MatchCake/MatchCake/blob/main/images/logo/Logo.svg?raw=true" width="60%" /></p>
<p align="center"><sub>Logo by Tarik El-Khateeb / Xanadu</sub></p>

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/MatchCake.svg?style=social)](https://github.com/MatchCake/MatchCake/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/MatchCake?style=social)](https://github.com/MatchCake/MatchCake/network/members)
[![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![downloads](https://img.shields.io/pypi/dm/MatchCake)](https://pypi.org/project/MatchCake)
[![PyPI version](https://img.shields.io/pypi/v/MatchCake)](https://pypi.org/project/MatchCake)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/MatchCake/MatchCake/blob/dev/LICENSE)
[![status](https://joss.theoj.org/papers/91f77a47cbd519daac9794a1d2144361/status.svg)](https://joss.theoj.org/papers/91f77a47cbd519daac9794a1d2144361)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22917735.svg)](https://doi.org/10.5281/zenodo.22917735)

![Tests Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/docs.yml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/MatchCake/MatchCake/branch/main/graph/badge.svg?token=Yz44IcMdVx)](https://codecov.io/github/MatchCake/MatchCake)


# Description

MatchCake is a Python package that provides a new PennyLane device for simulating a specific class of quantum
circuits called Matchgate circuits or matchcircuits. These circuits are made with matchgates, a class of restricted
quantum unitaries that are parity-preserving and operate on nearest-neighbor qubits. These constraints lead to
matchgates being classically simulable in polynomial time.

Additionally, this package provides quantum kernels made with [scikit-learn](https://scikit-learn.org/stable/) API allowing the
use matchcircuits as kernels in quantum machine learning algorithms. One way to use these kernels could be in a Support Vector
Machine (SVM).

Note that this package is built on PennyLane and PyTorch. This means that only the NumPy and PyTorch backends are compatible.
Other backends provided by Autoray, such as JAX and TensorFlow, are not supported.
We highly recommend using PyTorch as the backend when working with MatchCake.


# Requirements

| Requirement   | Supported                  |
|---------------|----------------------------|
| **Python**    | 3.11, 3.12, 3.13, 3.14     |
| **Platforms** | Linux, Windows             |

Every one of these Python versions is tested on Linux (x86-64 and aarch64) and on Windows on each pull
request, in [continuous integration](https://github.com/MatchCake/MatchCake/actions/workflows/tests.yml).
On Linux, a glibc based distribution is required. PyTorch publishes no musl wheels, so MatchCake cannot be
installed on Alpine.

macOS is not supported. MatchCake contains no platform specific code and may well run there, but it
is not tested. If you need macOS support, please
[open an issue](https://github.com/MatchCake/MatchCake/issues/new/choose) and we will look into
adding it to the continuous integration matrix.


# Installation

MatchCake is published on PyPI as `matchcake`.

### Set up a project or an environment

Install MatchCake into a virtual environment and not into the system interpreter, which recent Linux
distributions refuse with `error: externally-managed-environment`. If you do not have one yet, create it:

```bash
uv init myproject && cd myproject                    # uv
poetry new myproject && cd myproject                 # poetry (then cap requires-python, see below)
python -m venv .venv && source .venv/bin/activate    # plain virtual environment
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

### Install MatchCake

| Method     | Commands                                                          |
|------------|-------------------------------------------------------------------|
| **pip**    | `pip install matchcake`                                           |
| **uv**     | `uv add matchcake` in a uv project, or `uv pip install matchcake` |
| **poetry** | `poetry add matchcake` in a poetry project                        |
| **source** | `pip install "git+https://github.com/MatchCake/MatchCake@main"`   |

A few things useful to know during the installation:

- `uv add` and `poetry add` are project commands: they record a dependency in the `pyproject.toml` of the
  project you are standing in. In a directory without one they stop with
  `No pyproject.toml found in current directory or any parent directory`. Note that `uv add` also searches
  parent directories, so running it inside an unrelated project modifies that project. Use
  `uv pip install matchcake` when you only want the package in an environment.
- `poetry add matchcake` additionally requires your project to put an upper bound on its Python range, for
  example `requires-python = ">=3.11,<3.15"`. `poetry new` writes an open ended range such as `>=3.12`, and
  poetry then declines to resolve because `torchpfaffian` supports Python `<3.15` only.
- The `source` row needs `git` available on your `PATH`. It installs the latest release from the `main`
  branch; see below for the development branch.


### Last unstable version

To install the development branch, use the `@dev` reference:

```bash
pip install "git+https://github.com/MatchCake/MatchCake@dev"
```

The uv equivalent is `uv add "matchcake @ git+https://github.com/MatchCake/MatchCake@dev"`.


### PyTorch build and installation size

MatchCake depends on PyTorch, and on Linux the default PyTorch wheel on PyPI is the CUDA build. A plain
`pip install matchcake` therefore downloads PyTorch together with around twenty NVIDIA packages and occupies
roughly 6 GB, whether or not the machine has a GPU. This comes from how PyTorch is packaged, not from
MatchCake.

For a CPU only environment, install PyTorch from the PyTorch CPU index first, then install MatchCake on top.
The requirement is already satisfied at that point, so nothing pulls the CUDA build in, and the result is
about 1.6 GB with no NVIDIA packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matchcake
```

Swap the index for a specific CUDA version, for instance `https://download.pytorch.org/whl/cu128` for
CUDA 12.8 or `https://download.pytorch.org/whl/cu130` for CUDA 13.0.

With uv, declare the index in your own `pyproject.toml` and add `torch` alongside `matchcake`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

```bash
uv add matchcake torch
```

Listing `torch` explicitly is important: `[tool.uv.sources]` only redirects dependencies that your own project
declares, so a `torch` pulled in solely through `matchcake` still comes from PyPI.

MatchCake declares `cpu`, `cu128` and `cu130` extras, but they only select these indexes when MatchCake
itself is built from a clone, through `[tool.uv.sources]` in its `pyproject.toml`. That mechanism belongs to
uv and is not part of the metadata published to PyPI, so `pip install "matchcake[cpu]"` and
`uv add matchcake --extra cpu` resolve exactly the same PyTorch as no extra at all. Use the index based
recipe above. The extras are documented for contributors in
[CONTRIBUTING.md](https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md).


# Quick Usage Preview

## Quantum Circuit Simulation with MatchCake
```python
import matchcake as mc
import pennylane as qml
import numpy as np
from pennylane.ops.qubit.observables import BasisStateProjector

# Create a Non-Interacting Fermionic Device
nif_device = mc.NonInteractingFermionicDevice(wires=4)
initial_state = np.zeros(len(nif_device.wires), dtype=int)

# Define a quantum circuit
def circuit(params, wires, initial_state=None):
    qml.BasisState(initial_state, wires=wires)
    for i, even_wire in enumerate(wires[:-1:2]):
        idx = list(wires).index(even_wire)
        curr_wires = [wires[idx], wires[idx + 1]]
        mc.operations.CompRxRx(params, wires=curr_wires)
        mc.operations.CompRyRy(params, wires=curr_wires)
        mc.operations.CompRzRz(params, wires=curr_wires)
    for i, odd_wire in enumerate(wires[1:-1:2]):
        idx = list(wires).index(odd_wire)
        mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)

# Create a QNode
nif_qnode = qml.QNode(circuit, nif_device)
qml.draw_mpl(nif_qnode)(np.array([0.1, 0.2]), wires=nif_device.wires, initial_state=initial_state)

# Evaluate the QNode
expval = nif_qnode(np.random.random(2), wires=nif_device.wires, initial_state=initial_state)
print(f"Expectation value: {expval}")
```

## Mixed States, Partial Trace and Ancillas

**Only in this fork.** Everything in this section is on the `dev` branch of this fork and is not in upstream
MatchCake. Install it with `pip install "git+https://github.com/MarcoArmenta/MatchCake@dev"`.

Upstream MatchCake starts a circuit from a basis state or a product state and measures it at the end. This fork
adds what is needed to work with a part of the system, and to feed it back into another circuit:

| What you want to do | How |
| :--- | :--- |
| Start a circuit from any Gaussian state, pure or mixed | `GaussianStatePrep(covariance, wires=...)` |
| Trace out wires after a circuit | `device.partial_trace(wires)`, `device.reduced_state(wires)` |
| Read a reduced state: entropy, purity, normal modes | `MixedGaussianState` |
| Run another circuit on a reduced state | `state.state_prep()` in the next circuit |
| Trace out wires of a circuit that starts from a product state | same methods, the state is then *displaced* |
| Add ancilla wires to a state | `state.insert(index, ancilla)`, `state.append(ancilla)` |

A matchgate circuit keeps the state fermionic Gaussian, so a reduced state is fully described by the covariance
matrix of its wires (2m x 2m numbers for m wires) and nothing here builds a density matrix. The tutorial below
goes through these features one by one. The
[theory page](https://github.com/MarcoArmenta/MatchCake/blob/dev/docs/partial_trace_theory.md) has the math.

### 1. Trace out wires of a circuit

After a circuit has been applied to the device, `partial_trace` takes the wires to remove and `reduced_state`
the wires to keep. Both return a `MixedGaussianState` on the remaining wires, relabeled from 0.

```python
import matchcake as mc
import pennylane as qml
import numpy as np
import torch
from matchcake.operations.state_preparation import GaussianStatePrep, ProductState

nif_device = mc.NonInteractingFermionicDevice(wires=6)
nif_device.apply([
    qml.BasisState(np.array([1, 0, 1, 0, 1, 0]), wires=range(6)),
    mc.operations.CompRxRx(np.array([0.3, 0.7]), wires=[0, 1]),
    mc.operations.CompRyRy(np.array([0.2, 1.1]), wires=[2, 3]),
    mc.operations.CompRxRx(np.array([0.9, 0.4]), wires=[3, 4]),
    mc.operations.CompRyRy(np.array([0.5, 0.6]), wires=[4, 5]),
    mc.operations.CompRzRz(np.array([0.8, 0.1]), wires=[1, 2]),
])

# Trace out wires 4 and 5. The remaining wires are relabeled 0..3.
reduced = nif_device.partial_trace([4, 5])
# The same state, naming the wires to keep instead
reduced = nif_device.reduced_state([0, 1, 2, 3])

print(reduced.entropy())         # 0.2797, entanglement entropy between wires 0-3 and wires 4-5
print(reduced.purity)            # 0.8522, Tr(rho^2), which is 1 for a pure state
print(reduced.density_matrix())  # dense 16 x 16 matrix, for small systems only
```

`MixedGaussianState` is a container: it holds the state and gives what follows from its covariance matrix. It
does not run circuits. That is the job of the device, as in the next step.

### 2. Run another circuit on the reduced state

`state_prep()` returns the operation that prepares the state. Use it as the first operation of the next
circuit, like `qml.BasisState`. Expectation values, probabilities, samples and shots then work as for any
other circuit of the device.

```python
observable = qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(2) @ qml.PauliX(3)

@qml.qnode(qml.device("nif.qubit", wires=4))
def follow_up():
    reduced.state_prep()
    mc.operations.CompRxRx(np.array([0.1, 0.2]), wires=[1, 2])
    return qml.expval(observable), qml.probs(wires=[0, 1])

expval, probs = follow_up()  # -1.3900, [0.0005, 0.1170, 0.8754, 0.0070]

# The same circuit with a finite number of shots
expval, probs = qml.set_shots(follow_up, shots=20_000)()  # about -1.39, about [0.001, 0.113, 0.878, 0.008]
```

### 3. Start a circuit from a Gaussian state

A mixed state does not have to come from a circuit. `GaussianStatePrep` prepares the state of a covariance
matrix, and `MixedGaussianState.from_polarizations` builds simple ones: a product of single-wire mixtures.

```python
# One polarization per wire: +1 is |0>, -1 is |1>, 0 is maximally mixed
state = mc.MixedGaussianState.from_polarizations(np.array([1.0, 0.0, -1.0, 0.5]))
print(state.entropy(), state.purity)  # 1.2555, 0.3125

@qml.qnode(qml.device("nif.qubit", wires=4))
def circuit(covariance):
    GaussianStatePrep(covariance, wires=range(4))
    mc.operations.CompRxRx(np.array([0.3, 0.7]), wires=[0, 1])
    mc.operations.CompRyRy(np.array([0.2, 1.1]), wires=[2, 3])
    mc.operations.fSWAP(wires=[1, 2])
    return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

expval, probs = circuit(state.covariance_matrix)  # 0.8601, [0.1929, 0.7372, 0.0145, 0.0554]
```

The device keeps the state after a circuit, so a circuit and a trace can be chained without a QNode. A state
prepared on part of the wires leaves the other wires in |0>.

```python
device = mc.NonInteractingFermionicDevice(wires=4)
device.apply([
    state.state_prep(),
    mc.operations.CompRxRx(np.array([0.3, 0.7]), wires=[0, 1]),
    mc.operations.CompRyRy(np.array([0.2, 1.1]), wires=[2, 3]),
    mc.operations.fSWAP(wires=[1, 2]),
])
print(device.partial_trace([3]).entropy())  # 1.2036

# The 4-wire state on wires 1-4 of a larger device, wires 0 and 5 in |0>
larger_device = mc.NonInteractingFermionicDevice(wires=6)
larger_device.apply([state.state_prep(wires=[1, 2, 3, 4])])
```

### 4. Wires that are not next to each other

Keeping a contiguous block of wires, as above, gives the usual partial trace of the qubits. Keeping wires that
leave gaps does not: the reduced state that stays Gaussian is the *fermionic* partial trace, and it has to be
requested explicitly.

```python
nif_device.partial_trace([1, 3])  # ValueError: ... not a qubit partial trace ...
separated = nif_device.partial_trace([1, 3], fermionic=True)
print(separated.entropy())  # 0.9246
```

The fermionic partial trace is the usual partial trace of the state whose traced-out wires were first moved
behind the kept ones with `fSWAP` gates. So "move the wires to the end with `fSWAP`, then discard them" is always
exact.

### 5. Trace out an ancilla prepared in a product state

The circuit can start from any product state, for instance a system in a basis state next to an ancilla in a
random product state. Put the wires to trace out last.

```python
rng = np.random.default_rng(0)
# Wires 0-2 are the system, in |010>. Wires 3-4 are the ancilla, in a random product state.
system = np.eye(2)[[0, 1, 0]]
ancilla = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
ancilla /= np.linalg.norm(ancilla, axis=-1, keepdims=True)

ancilla_device = mc.NonInteractingFermionicDevice(wires=5)
ancilla_device.apply([
    ProductState(np.concatenate([system, ancilla]), wires=range(5)),
    mc.operations.CompRxRx(np.array([0.3, 0.7]), wires=[2, 3]),
    mc.operations.CompRyRy(np.array([0.2, 1.1]), wires=[3, 4]),
    mc.operations.CompRzRz(np.array([0.8, 0.1]), wires=[1, 2]),
    mc.operations.CompRxRx(np.array([0.9, 0.4]), wires=[2, 3]),
])

system_state = ancilla_device.partial_trace([3, 4])
print(system_state.is_displaced)  # True
print(system_state.entropy())     # 0.3598

@qml.qnode(qml.device("nif.qubit", wires=3))
def system_circuit():
    system_state.state_prep()
    mc.operations.CompRyRy(np.array([0.5, 0.6]), wires=[1, 2])
    return qml.expval(qml.PauliX(2) + 0.5 * qml.PauliZ(0) @ qml.PauliZ(1))

print(system_circuit())  # -0.5757
```

Such a state is called *displaced*: operators with an odd number of X and Y factors, like `qml.PauliX(2)`, have
a nonzero expectation value in it, and the state carries 2m more numbers, its `displacement`. Everything else
works the same. For a displaced state, the kept block must start at the first wire for the trace to be the usual
partial trace. Tracing out wires in front of it also needs `fermionic=True`.

### 6. Add ancilla wires to a state

`insert` adds ancilla wires at a position of the wire list, as `list.insert` does, and `append` adds them behind
the last wire. Both return a new state, the tensor product of the state with the ancilla.

```python
fresh = rng.normal(size=(1, 2)) + 1j * rng.normal(size=(1, 2))
fresh /= np.linalg.norm(fresh, axis=-1, keepdims=True)

in_front = system_state.insert(0, fresh)  # the ancilla is wire 0, the system is on wires 1-3
between = system_state.insert(2, [1])     # an ancilla in |1> between the system wires 1 and 2
behind = system_state.append([0, 0])      # two ancillas in |00> on wires 3-4
print(in_front.num_wires, between.num_wires, behind.num_wires)  # 4 4 5

# One more round: couple the fresh ancilla to the system, move it to the last wire, and discard it
round_device = mc.NonInteractingFermionicDevice(wires=4)
round_device.apply([
    in_front.state_prep(),
    mc.operations.CompRxRx(np.array([0.4, 0.2]), wires=[0, 1]),
    mc.operations.fSWAP(wires=[0, 1]),
    mc.operations.fSWAP(wires=[1, 2]),
    mc.operations.fSWAP(wires=[2, 3]),
])
system_state = round_device.partial_trace([3])
print(system_state.entropy())  # 0.6094
```

The ancilla is given as bits or a `qml.BasisState` for a computational-basis state, and as amplitudes of shape
`(k, 2)` or a `ProductState` for any product state of `k` wires. Where it can go depends on its state.

- An ancilla in a computational-basis state can be inserted at any position of any state.
- An ancilla with superposed qubits can be inserted where the wires in front of it are in a pure state. This is
  always the case at position 0. Behind wires in a mixed state, the tensor product is no longer a Gaussian state
  and `insert` raises a `ValueError`. To bring such an ancilla next to a wire in the middle, insert it at
  position 0 and move it with `fSWAP` gates, as in the example.

### 7. Gradients

With PyTorch parameters, the reduced state and everything computed from it are differentiable, including the
entropy and a circuit that starts from the reduced state.

```python
params = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
grad_device = mc.NonInteractingFermionicDevice(wires=4)
grad_device.apply([
    qml.BasisState(np.array([1, 0, 1, 0]), wires=range(4)),
    mc.operations.CompRxRx(params, wires=[0, 1]),
    mc.operations.CompRyRy(params, wires=[1, 2]),
    mc.operations.CompRxRx(params, wires=[2, 3]),
])
half = grad_device.partial_trace([2, 3])

# Entanglement entropy between wires 0-1 and wires 2-3, and its gradient
entropy = half.entropy()
entropy.backward(retain_graph=True)
print(entropy.item(), params.grad)  # 0.3389, [0.0370, 0.5396]

# A circuit on the reduced state, differentiated through the covariance matrix
@qml.qnode(qml.device("nif.qubit", wires=2), interface="torch", diff_method="backprop")
def half_circuit(covariance):
    GaussianStatePrep(covariance, wires=[0, 1])
    mc.operations.CompRyRy(np.array([0.5, 0.6]), wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

params.grad = None
value = half_circuit(half.covariance_matrix)
value.backward()
print(value.item(), params.grad)  # -0.6365, [0.0009, 0.5192]
```

### 8. The pure states inside a mixed state

A reduced state is a mixture of a few pure Gaussian states, at most 2^k of them for k traced-out wires.
`weights` holds their probabilities and `pure_states()` iterates over them, each one as a basis state followed by
a transition matrix that the device can prepare. This is a way to look inside the state or to cross-check a
result. Running a circuit does not need it: the device evolves the covariance matrix in a single execution.

```python
print(reduced.num_pure_states, reduced.weights)  # 2, [0.0804, 0.9196]

total = 0.0
check_device = mc.NonInteractingFermionicDevice(wires=reduced.ensemble_wires)
for weight, basis_state, transition in reduced.pure_states():
    value = check_device.execute_generator(
        [basis_state, transition, mc.operations.CompRxRx(np.array([0.1, 0.2]), wires=[1, 2])],
        observable=reduced.ensemble_observable(observable),
        output_type="expval",
        reset=True,
    )
    total += float(weight) * float(value)
print(total)  # -1.3900, the value of step 2
```

### Limits

- The first circuit must start from a state with a covariance matrix: a basis state, a `ProductState` or a
  `GaussianStatePrep`.
- The usual partial trace of the qubits needs the kept wires to be a contiguous block, which must start at the
  first wire when the state is displaced. Every other case is the fermionic partial trace and needs
  `fermionic=True`.
- An ancilla with superposed qubits can only be added behind wires that are in a pure state.
- A covariance matrix that is not the one of a state is rejected when its normal modes are computed: no mode can
  have a polarization larger than one.

## Data Classification with MatchCake

```python
from matchcake.ml.kernels import FermionicPQCKernel
from matchcake.ml.visualisation import ClassificationVisualizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load the iris dataset
X, y = datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the model
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('kernel', FermionicPQCKernel(n_qubits=4, rotations="X,Z").freeze()),
    ('classifier', SVC(kernel='precomputed')),
])
pipeline.fit(x_train, y_train)

# Evaluate the model
test_accuracy = pipeline.score(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Visualize the classification
viz = ClassificationVisualizer(x=X, n_pts=1_000)
viz.plot_2d_decision_boundaries(model=pipeline, y=y, show=True)

```


# Tutorials
- [MatchCake Basics](https://github.com/MatchCake/MatchCake/blob/main/tutorials/matchcake_basics.ipynb)
- [Compute Expectation Values with MatchCake](https://github.com/MatchCake/MatchCake/blob/main/tutorials/expectation_values.ipynb)
- [Sampling with MatchCake](https://github.com/MatchCake/MatchCake/blob/main/tutorials/sampling.ipynb)
- [Iris Classification with MatchCake](https://github.com/MatchCake/MatchCake/blob/main/tutorials/iris_classification.ipynb)
- [Nystroem Kernel Approximation](https://github.com/MatchCake/MatchCake/blob/main/tutorials/nystroem_kernel_approximation.ipynb)


# For Developers

To contribute to the development of MatchCake, please refer to the [contributing guidelines](https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md).



# Notes
- This package is still in development and some features may not be available yet.
- The documentation is still in development and may not be complete yet.



# About
This work was supported by the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec
through its Research Chair in Quantum Computing, an NSERC Discovery grant, and the Canada First Research Excellence
Fund.


# Important Links
- Documentation at [https://MatchCake.github.io/MatchCake/](https://MatchCake.github.io/MatchCake/).
- Github at [https://github.com/MatchCake/MatchCake/](https://github.com/MatchCake/MatchCake/).




# Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/MatchCake/MatchCake/issues/new/choose)



# License
[Apache License 2.0](https://github.com/MatchCake/MatchCake/blob/dev/LICENSE)



# Citations

Software, archived on [Zenodo](https://doi.org/10.5281/zenodo.22917735) (this DOI resolves to the latest version):
```
@software{matchcake_Gince2026,
  title={MatchCake: A Python Simulator for Non-Interacting Fermionic Quantum Circuits with Machine Learning Applications},
  author={Gince, Jérémie},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22917735},
  url={https://doi.org/10.5281/zenodo.22917735},
}
```

## Fermionic Machine Learning Paper

[Fermionic Machine Learning](https://ieeexplore.ieee.org/document/10821385) is a work presented at the 2024 IEEE
International Conference on Quantum Computing and Engineering (QCE). The paper compares unconstrained quantum kernel
methods with constraint-based kernels derived from matchgate (free-fermionic) circuits, and benchmarks their
performance on supervised classification tasks. All free-fermionic kernels considered in this work were simulated
using MatchCake.

[IEEE Xplore paper](https://ieeexplore.ieee.org/document/10821385):
```
@INPROCEEDINGS{10821385,
  author={Gince, Jérémie and Pagé, Jean-Michel and Armenta, Marco and Sarkar, Ayana and Kourtis, Stefanos},
  booktitle={2024 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  title={Fermionic Machine Learning},
  year={2024},
  volume={01},
  number={},
  pages={1672-1678},
  keywords={Runtime;Quantum entanglement;Computational modeling;Benchmark testing;Rendering (computer graphics);Hardware;Kernel;Integrated circuit modeling;Quantum circuit;Standards;Quantum machine learning;quantum kernel methods;matchgate circuits;fermionic quantum computation;data classification},
  doi={10.1109/QCE60285.2024.00195}
}
```


[ArXiv paper](https://arxiv.org/abs/2404.19032):
```
@misc{gince2024fermionic,
      title={Fermionic Machine Learning},
      author={Jérémie Gince and Jean-Michel Pagé and Marco Armenta and Ayana Sarkar and Stefanos Kourtis},
      year={2024},
      eprint={2404.19032},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
