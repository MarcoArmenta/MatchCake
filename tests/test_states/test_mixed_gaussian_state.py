import warnings
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import BatchHamiltonian, MixedGaussianState, NonInteractingFermionicDevice
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.operations.state_preparation import ProductState
from matchcake.utils.covariance import block_diagonal_covariance
from matchcake.utils.majorana import get_majorana

from ..configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)

N_SHOTS = 20_000


class TestMixedGaussianState:
    @staticmethod
    def brick_wall(wires, depth: int, rng: np.random.Generator):
        """Random layers of ``CompRxRx``, ``CompRyRy``, ``CompRzRz`` and an ``fSWAP`` on consecutive wires.

        :param wires: The wires of the circuit.
        :type wires: Iterable[int]
        :param depth: Number of layers.
        :type depth: int
        :param rng: Random generator drawing the angles.
        :type rng: np.random.Generator
        :return: The operations.
        :rtype: List[Operation]
        """
        wires = list(wires)
        operations = []
        for _ in range(depth):
            for position in range(len(wires) - 1):
                pair = [wires[position], wires[position + 1]]
                operations.append(CompRxRx(rng.normal(size=2), wires=pair))
                operations.append(CompRyRy(rng.normal(size=2), wires=pair))
                operations.append(CompRzRz(rng.normal(size=2), wires=pair))
            operations.append(fSWAP(wires=[wires[0], wires[1]]))
        return operations

    @staticmethod
    def dense_state(basis_state, operations, n_wires: int) -> np.ndarray:
        """Run the circuit on ``default.qubit`` and return the state vector.

        :param basis_state: Initial computational-basis state.
        :type basis_state: np.ndarray
        :param operations: Operations of the circuit, built outside of any queuing context.
        :type operations: List[Operation]
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: The state vector.
        :rtype: np.ndarray
        """
        device = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(device)
        def circuit():
            qml.BasisState(np.asarray(basis_state), wires=range(n_wires))
            for operation in operations:
                qml.apply(operation)
            return qml.state()

        return np.asarray(circuit())

    @staticmethod
    def dense_covariance(density_matrix: np.ndarray, n_wires: int) -> np.ndarray:
        """Brute-force Majorana covariance matrix ``i Tr[rho c_mu c_nu]`` of a density matrix.

        :param density_matrix: Density matrix of shape ``(2^n, 2^n)``.
        :type density_matrix: np.ndarray
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: Real antisymmetric matrix of shape ``(2n, 2n)``.
        :rtype: np.ndarray
        """
        covariance = np.zeros((2 * n_wires, 2 * n_wires), dtype=complex)
        for mu in range(2 * n_wires):
            for nu in range(2 * n_wires):
                if mu != nu:
                    covariance[mu, nu] = 1j * np.trace(
                        density_matrix @ get_majorana(mu, n_wires) @ get_majorana(nu, n_wires)
                    )
        return covariance.real

    @staticmethod
    def dense_unitary(operations, n_wires: int) -> np.ndarray:
        """Dense unitary of a circuit on ``range(n_wires)``.

        :param operations: Operations of the circuit.
        :type operations: List[Operation]
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: Unitary matrix of shape ``(2^n, 2^n)``.
        :rtype: np.ndarray
        """
        return np.asarray(qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(n_wires)))

    @staticmethod
    def reduced_from_device(device: NonInteractingFermionicDevice, traced_wires) -> MixedGaussianState:
        """Reduced state of the device on the wires not in ``traced_wires``, relabeled from zero.

        The covariance matrix of the reduced state is the principal submatrix of the device's covariance matrix on
        the Majorana indices of the kept wires.

        :param device: A device whose circuit has been applied.
        :type device: NonInteractingFermionicDevice
        :param traced_wires: Wires to trace out.
        :type traced_wires: Iterable[int]
        :return: The reduced state.
        :rtype: MixedGaussianState
        """
        kept = [wire for wire in range(len(device.wires)) if wire not in traced_wires]
        indices = MixedGaussianState.majorana_indices(kept)
        covariance_matrix: Any = device.covariance_matrix
        covariance = covariance_matrix[..., indices[:, None], indices[None, :]]
        return MixedGaussianState(covariance, source_wires=kept)

    @staticmethod
    def random_product_amplitudes(n_wires: int, rng: np.random.Generator) -> np.ndarray:
        """Random normalized single-qubit amplitudes for every wire.

        :param n_wires: Number of wires.
        :type n_wires: int
        :param rng: Random generator drawing the amplitudes.
        :type rng: np.random.Generator
        :return: Complex array of shape ``(n_wires, 2)``.
        :rtype: np.ndarray
        """
        amplitudes = rng.normal(size=(n_wires, 2)) + 1j * rng.normal(size=(n_wires, 2))
        return amplitudes / np.linalg.norm(amplitudes, axis=-1, keepdims=True)

    @staticmethod
    def dense_product_state(amplitudes, operations, n_wires: int) -> np.ndarray:
        """Apply the dense unitary of a circuit to the tensor product of single-qubit states.

        :param amplitudes: Single-qubit amplitudes of shape ``(n_wires, 2)``.
        :type amplitudes: np.ndarray
        :param operations: Operations of the circuit.
        :type operations: List[Operation]
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: The state vector.
        :rtype: np.ndarray
        """
        psi = np.ones(1, dtype=complex)
        for qubit_amplitudes in np.asarray(amplitudes):
            psi = np.kron(psi, qubit_amplitudes)
        unitary = np.asarray(qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(n_wires)))
        return unitary @ psi

    @staticmethod
    def displaced_from_device(device: NonInteractingFermionicDevice, kept) -> MixedGaussianState:
        """Displaced reduced state of the device on the wires ``kept``, relabeled from zero.

        The covariance matrix and the displacement are read from the extended covariance matrix of the device, on
        the Majorana indices of the kept wires and on its parity index.

        :param device: A device whose circuit has been applied to a product state.
        :type device: NonInteractingFermionicDevice
        :param kept: Wires to keep.
        :type kept: List[int]
        :return: The reduced state.
        :rtype: MixedGaussianState
        """
        indices = MixedGaussianState.majorana_indices(kept)
        extended: Any = device.extended_covariance_matrix
        covariance = extended[..., indices[:, None], indices[None, :]]
        return MixedGaussianState(covariance, displacement=extended[..., indices, -1], source_wires=kept)

    @staticmethod
    def dense_ancilla(amplitudes) -> np.ndarray:
        """Dense density matrix of the tensor product of single-qubit pure states.

        :param amplitudes: Single-qubit amplitudes of shape ``(n_wires, 2)``.
        :type amplitudes: np.ndarray
        :return: Density matrix of shape ``(2^n, 2^n)``.
        :rtype: np.ndarray
        """
        psi = np.ones(1, dtype=complex)
        for qubit_amplitudes in np.asarray(amplitudes):
            psi = np.kron(psi, qubit_amplitudes)
        return np.outer(psi, psi.conj())

    @staticmethod
    def dense_insert(density: np.ndarray, ancilla_density: np.ndarray, index: int) -> np.ndarray:
        """Qubit tensor product of two density matrices, with the wires of the second at position ``index``.

        :param density: Density matrix of ``m`` wires.
        :type density: np.ndarray
        :param ancilla_density: Density matrix of the ``k`` inserted wires.
        :type ancilla_density: np.ndarray
        :param index: Position of the first inserted wire among the ``m + k`` wires.
        :type index: int
        :return: Density matrix of shape ``(2^(m + k), 2^(m + k))``.
        :rtype: np.ndarray
        """
        n_wires = int(np.log2(density.shape[0]))
        n_ancillas = int(np.log2(ancilla_density.shape[0]))
        total = n_wires + n_ancillas
        joint = np.kron(density, ancilla_density).reshape([2] * (2 * total))
        order = list(range(index)) + list(range(n_wires, total)) + list(range(index, n_wires))
        return joint.transpose(order + [total + axis for axis in order]).reshape(2**total, 2**total)

    @pytest.fixture
    def displaced_case(self):
        """A 5-wire circuit on a random product state reduced to the wires ``[0, 1, 2]`` with its dense reference."""
        rng = np.random.default_rng(TEST_SEED + 11)
        n_wires = 5
        amplitudes = self.random_product_amplitudes(n_wires, rng)
        operations = self.brick_wall(range(n_wires), 2, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires))] + operations)
        state = self.displaced_from_device(device, [0, 1, 2])
        psi = self.dense_product_state(amplitudes, operations, n_wires)
        reduced_density = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [3, 4]))
        follow_up = self.brick_wall(range(3), 1, rng)
        unitary = self.dense_unitary(follow_up, 3)
        evolved_density = unitary @ reduced_density @ unitary.conj().T
        return state, reduced_density, follow_up, evolved_density

    @pytest.fixture
    def reduced_case(self):
        """A 5-wire pure state reduced to the contiguous block ``[1, 2, 3]`` with its dense reference."""
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 5
        basis_state = np.array([1, 0, 1, 1, 0])
        operations = self.brick_wall(range(n_wires), 2, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([qml.BasisState(basis_state, wires=range(n_wires))] + operations)
        state = self.reduced_from_device(device, [0, 4])
        psi = self.dense_state(basis_state, operations, n_wires)
        reduced_density = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [0, 4]))
        follow_up = self.brick_wall(range(3), 1, rng)
        unitary = self.dense_unitary(follow_up, 3)
        evolved_density = unitary @ reduced_density @ unitary.conj().T
        return state, reduced_density, follow_up, evolved_density

    def test_init_rejects_odd_or_non_square_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            MixedGaussianState(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="shape"):
            MixedGaussianState(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="shape"):
            MixedGaussianState(np.zeros((2, 2, 4, 4)))

    def test_init_rejects_non_antisymmetric_matrix(self):
        with pytest.raises(ValueError, match="antisymmetric"):
            MixedGaussianState(np.eye(4))

    def test_init_rejects_wrong_number_of_wires(self):
        with pytest.raises(ValueError, match="Expected 2 wires"):
            MixedGaussianState(np.zeros((4, 4)), wires=[0, 1, 2])

    def test_init_rejects_non_consecutive_wires(self):
        with pytest.raises(ValueError, match="consecutive"):
            MixedGaussianState(np.zeros((4, 4)), wires=[0, 2])
        with pytest.raises(ValueError, match="consecutive"):
            MixedGaussianState(np.zeros((4, 4)), wires=["a", "b"])

    def test_init_accepts_shifted_consecutive_wires(self):
        state = MixedGaussianState(np.zeros((4, 4)), wires=[3, 4])
        assert state.wires.tolist() == [3, 4]
        assert state.source_wires.tolist() == [3, 4]

    def test_defaults(self):
        state = MixedGaussianState(np.zeros((6, 6)))
        assert state.wires.tolist() == [0, 1, 2]
        assert state.num_wires == 3
        assert state.batch_shape == ()
        assert state.atol == MixedGaussianState.DEFAULT_ATOL
        assert state.device_kwargs == {}

    def test_repr(self):
        state = MixedGaussianState.from_polarizations([1.0, 0.0])
        assert repr(state) == "MixedGaussianState(num_wires=2, batch_shape=(), num_mixed_modes=1, num_pure_states=2)"

    def test_majorana_indices(self):
        np.testing.assert_array_equal(MixedGaussianState.majorana_indices([0, 2]), [0, 1, 4, 5])
        np.testing.assert_array_equal(MixedGaussianState.majorana_indices([]), np.zeros(0, dtype=int))

    def test_from_polarizations_identity_sptm(self):
        polarizations = np.array([0.5, -1.0, 0.0])
        state = MixedGaussianState.from_polarizations(polarizations)
        np.testing.assert_allclose(
            state.covariance_matrix,
            block_diagonal_covariance(polarizations),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            np.sort(np.abs(state.mode_polarizations)),
            [0.0, 0.5, 1.0],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_from_polarizations_with_sptm(self):
        polarizations = np.array([0.5, -0.2])
        sptm = SingleParticleTransitionMatrixOperation.random_params(wires=[0, 1], seed=TEST_SEED).real
        state = MixedGaussianState.from_polarizations(polarizations, sptm=sptm, atol=1e-6)
        expected = sptm.T @ block_diagonal_covariance(polarizations) @ sptm
        np.testing.assert_allclose(
            state.covariance_matrix, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert state.atol == 1e-6

    def test_pure_state_has_single_member(self):
        state = MixedGaussianState.from_polarizations([1.0, -1.0, 1.0])
        assert state.is_pure
        assert state.mixed_modes == []
        assert state.num_mixed_modes == 0
        assert state.num_pure_states == 1
        assert state.basis_states.shape == (1, 3)
        np.testing.assert_allclose(state.weights, [1.0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        # The normal modes may be permuted by the decomposition, but the state is the basis state |010>.
        expected_probs = np.zeros(8)
        expected_probs[0b010] = 1.0
        np.testing.assert_allclose(state.probs(), expected_probs, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(state.purity, 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(state.entropy(), 0.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_mixed_modes_basis_states_and_weights(self):
        state = MixedGaussianState.from_polarizations([0.6, -1.0, -0.2])
        assert not state.is_pure
        assert state.num_mixed_modes == 2
        assert state.num_pure_states == 4
        assert state.basis_states.shape == (4, 3)
        # Every member is a distinct basis state of the normal modes with the product weights.
        assert len({tuple(bits) for bits in state.basis_states}) == 4
        expected_weights = [0.8 * 0.4, 0.8 * 0.6, 0.2 * 0.4, 0.2 * 0.6]
        np.testing.assert_allclose(
            np.sort(state.weights), np.sort(expected_weights), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        # The covariance matrix is already block diagonal, so the state is diagonal in the computational basis.
        bits = np.array([[(index >> shift) & 1 for shift in (2, 1, 0)] for index in range(8)])
        polarizations = np.array([0.6, -1.0, -0.2])
        expected_probs = np.prod(0.5 * (1.0 + (1 - 2 * bits) * polarizations), axis=-1)
        np.testing.assert_allclose(
            state.probs(), expected_probs, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            state.purity, (1 + 0.36) / 2 * (1 + 0.04) / 2, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_atol_controls_purity_detection(self):
        polarizations = [1.0 - 1e-5, 0.3]
        assert MixedGaussianState.from_polarizations(polarizations, atol=1e-8).num_mixed_modes == 2
        assert MixedGaussianState.from_polarizations(polarizations, atol=1e-4).num_mixed_modes == 1

    def test_entropy_matches_binary_entropy_and_base(self):
        state = MixedGaussianState.from_polarizations([0.0, 1.0, 0.5])
        expected = np.log(2) + (-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
        np.testing.assert_allclose(state.entropy(), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(
            state.entropy(base=2), expected / np.log(2), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_batched_polarizations_union_of_mixed_modes(self):
        polarizations = np.array([[1.0, 0.5], [-0.3, 1.0]])
        state = MixedGaussianState.from_polarizations(polarizations)
        assert state.batch_shape == (2,)
        # Each element has one mixed mode; the union over the batch decides the members.
        assert state.num_mixed_modes == 1
        assert state.basis_states.shape == (2, 2, 2)
        np.testing.assert_allclose(np.sum(state.weights, axis=-1), [1.0, 1.0], atol=ATOL_SCALAR_COMPARISON)
        np.testing.assert_array_equal(state.is_pure, [False, False])
        assert state.entropy().shape == (2,)
        assert state.purity.shape == (2,)

    def test_sptm_operation_and_pure_state_operations(self):
        state = MixedGaussianState.from_polarizations([0.5, 1.0])
        operation = state.sptm_operation
        assert isinstance(operation, SingleParticleTransitionMatrixOperation)
        assert operation.wires.tolist() == [0, 1]
        np.testing.assert_allclose(
            operation.sptm_matrix(), state.sptm, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        prep, sptm_op = state.pure_state_operations(1)
        assert isinstance(prep, ProductState)
        np.testing.assert_array_equal(prep.basis_state_bits(), state.basis_states[1])
        assert isinstance(sptm_op, SingleParticleTransitionMatrixOperation)

    def test_pure_states_iterates_members(self):
        state = MixedGaussianState.from_polarizations([0.5, 1.0, -0.1])
        members = list(state.pure_states())
        assert len(members) == state.num_pure_states == 4
        weights = np.array([float(weight) for weight, _, _ in members])
        np.testing.assert_allclose(weights, state.weights, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        for index, (_, prep, sptm_op) in enumerate(members):
            np.testing.assert_array_equal(prep.basis_state_bits(), state.basis_states[index])
            assert sptm_op is members[0][2]

    def test_ensemble_operations_unbatched(self):
        state = MixedGaussianState.from_polarizations([0.5, 1.0, -0.1])
        prep, sptm_op = state.ensemble_operations()
        assert prep.batch_size == 4
        np.testing.assert_array_equal(prep.basis_state_bits(), state.basis_states)
        assert sptm_op.batch_size is None

    def test_ensemble_operations_batched_repeats_sptm(self):
        polarizations = np.array([[1.0, 0.5], [-0.3, 1.0]])
        state = MixedGaussianState.from_polarizations(polarizations)
        num_members = state.num_pure_states
        prep, sptm_op = state.ensemble_operations()
        assert prep.batch_size == 2 * num_members
        assert sptm_op.batch_size == 2 * num_members
        sptm = np.asarray(sptm_op.sptm_matrix())
        for batch_index in range(2):
            for member in range(num_members):
                np.testing.assert_allclose(
                    sptm[batch_index * num_members + member], state.sptm[batch_index], atol=ATOL_MATRIX_COMPARISON
                )

    def test_ensemble_reconstructs_density_matrix(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        n_wires = state.num_wires
        reconstructed = np.zeros((2**n_wires, 2**n_wires), dtype=complex)
        for weight, prep, sptm_op in state.pure_states():
            unitary = np.asarray(
                SingleParticleTransitionMatrixOperation.to_unitary_matrix(torch.as_tensor(sptm_op.sptm_matrix()))
            )
            vector = unitary @ prep.state_vector().reshape(-1)
            reconstructed += float(weight) * np.outer(vector, vector.conj())
        np.testing.assert_allclose(
            reconstructed, reduced_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_density_matrix_matches_qubit_partial_trace(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        np.testing.assert_allclose(
            state.density_matrix(), reduced_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_density_matrix_batched_and_backend(self):
        polarizations = torch.tensor([[1.0, 0.0], [-1.0, 0.4]], dtype=torch.float64)
        state = MixedGaussianState.from_polarizations(polarizations)
        density = state.density_matrix()
        assert isinstance(density, torch.Tensor)
        assert density.shape == (2, 4, 4)
        expected_first = np.diag([0.5, 0.5, 0.0, 0.0])
        expected_second = np.diag([0.0, 0.0, 0.7, 0.3])
        np.testing.assert_allclose(density[0].numpy(), expected_first, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(density[1].numpy(), expected_second, atol=ATOL_MATRIX_COMPARISON)

    def test_entropy_and_purity_match_dense(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        eigenvalues = np.linalg.eigvalsh(reduced_density)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        np.testing.assert_allclose(
            state.entropy(),
            -np.sum(eigenvalues * np.log(eigenvalues)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            state.purity,
            np.trace(reduced_density @ reduced_density).real,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_covariance_matrix_matches_dense(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        np.testing.assert_allclose(
            state.covariance_matrix,
            self.dense_covariance(reduced_density, state.num_wires),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_extended_covariance_matrix_has_zero_displacement(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        extended = state.extended_covariance_matrix
        assert extended.shape == (5, 5)
        np.testing.assert_allclose(extended[:4, :4], state.covariance_matrix, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(extended[4, :], 0.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(extended[:, 4], 0.0, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_expval_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        observable = qml.PauliX(0) @ qml.PauliY(1) + 0.5 * qml.PauliZ(2) + 0.3 * qml.PauliX(1) @ qml.PauliX(2)
        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
        value = state.execute(follow_up, observable, "expval", mode=mode)
        np.testing.assert_allclose(value, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_batch_hamiltonian_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        terms = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(1) @ qml.PauliX(2)]
        coefficients = [0.2, -0.7]
        observable = BatchHamiltonian(coefficients, terms)
        expected = [
            coefficient * np.trace(evolved_density @ qml.matrix(term, wire_order=range(3))).real
            for coefficient, term in zip(coefficients, terms)
        ]
        value = state.execute(follow_up, observable, "expval", mode=mode)
        np.testing.assert_allclose(value, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_probs_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        probabilities = state.execute(follow_up, output_type="probs", mode=mode)
        np.testing.assert_allclose(
            probabilities, np.diag(evolved_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_marginal_probs_matches_dense(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        marginal = np.asarray(qml.math.partial_trace(evolved_density, [1]))
        probabilities = state.execute(follow_up, output_type="probs", mode=mode, wires=[0, 2])
        np.testing.assert_allclose(
            probabilities, np.diag(marginal).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_samples_follow_probabilities(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        np.random.seed(TEST_SEED)
        samples = state.execute(follow_up, output_type="samples", mode=mode, shots=N_SHOTS)
        assert samples.shape == (N_SHOTS, 3)
        indices = samples @ (2 ** np.arange(3)[::-1])
        frequencies = np.bincount(indices, minlength=8) / N_SHOTS
        np.testing.assert_allclose(frequencies, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_execute_shots_expval_and_probs_are_estimated_from_samples(self, reduced_case, mode):
        state, _, follow_up, evolved_density = reduced_case
        np.random.seed(TEST_SEED)
        observable = qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2)
        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
        value = state.execute(follow_up, observable, "expval", mode=mode, shots=N_SHOTS)
        np.testing.assert_allclose(value, expected, atol=ATOL_APPROX_COMPARISON)
        probabilities = state.execute(follow_up, output_type="probs", mode=mode, shots=N_SHOTS)
        np.testing.assert_allclose(probabilities, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)

    def test_execute_shots_probs_batched_state(self):
        state = MixedGaussianState.from_polarizations(np.array([[1.0, 0.5], [-1.0, 1.0]]))
        np.random.seed(TEST_SEED)
        probabilities = state.execute([], output_type="probs", mode="parallel", shots=N_SHOTS)
        assert probabilities.shape == (2, 4)
        np.testing.assert_allclose(probabilities[0], [0.75, 0.25, 0.0, 0.0], atol=ATOL_APPROX_COMPARISON)
        np.testing.assert_allclose(probabilities[1], [0.0, 0.0, 1.0, 0.0], atol=ATOL_APPROX_COMPARISON)
        value = state.execute([], qml.PauliZ(1), "expval", mode="direct", shots=N_SHOTS)
        np.testing.assert_allclose(value, [0.5, 1.0], atol=ATOL_APPROX_COMPARISON)

    def test_execute_returns_none_without_output_type(self, reduced_case):
        state, _, follow_up, _ = reduced_case
        assert state.execute(follow_up) is None

    def test_execute_rejects_unknown_output_type_and_mode(self, reduced_case):
        state, _, follow_up, _ = reduced_case
        with pytest.raises(ValueError, match="Output type"):
            state.execute(follow_up, output_type="star_state")
        with pytest.raises(ValueError, match="execution mode"):
            state.execute(follow_up, output_type="probs", mode="fastest")

    def test_execute_rejects_state_preparations_and_foreign_wires(self, reduced_case):
        state, _, _, _ = reduced_case
        with pytest.raises(ValueError, match="state preparation"):
            state.execute([qml.BasisState(np.zeros(3, dtype=int), wires=range(3))], output_type="probs")
        with pytest.raises(ValueError, match="not wires of the state"):
            state.execute([CompRxRx(np.zeros(2), wires=[2, 3])], output_type="probs")

    def test_execute_device_kwargs_override(self, reduced_case):
        state, _, follow_up, evolved_density = reduced_case
        state.device_kwargs["r_dtype"] = torch.float64
        probabilities = state.execute(
            follow_up, output_type="probs", mode="direct", device_kwargs={"pfaffian_chunk_size": 2}
        )
        np.testing.assert_allclose(
            probabilities, np.diag(evolved_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_expval_and_probs_without_circuit(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        observable = qml.PauliY(0) @ qml.PauliZ(1) @ qml.PauliX(2)
        expected = np.trace(reduced_density @ qml.matrix(observable, wire_order=range(3))).real
        np.testing.assert_allclose(
            state.expval(observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        np.testing.assert_allclose(
            state.probs(), np.diag(reduced_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert state.execute([], observable, "expval") == pytest.approx(expected, abs=ATOL_SCALAR_COMPARISON)

    def test_expval_projector(self, reduced_case):
        state, reduced_density, _, _ = reduced_case
        projector = qml.Projector(np.array([1, 0]), wires=[0, 2])
        marginal = np.asarray(qml.math.partial_trace(reduced_density, [1]))
        expected = np.diag(marginal).real[2]
        np.testing.assert_allclose(
            state.expval(projector), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_expval_rejects_unsupported_observable(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        with pytest.raises(NotImplementedError, match="cannot be computed"):
            state.expval(qml.Hermitian(np.eye(2), wires=[0]) @ qml.Hermitian(np.eye(2), wires=[1]))

    def test_sample_requires_shots(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        with pytest.raises(ValueError, match="shots"):
            state.sample(None)

    def test_sample_batched_shape(self):
        state = MixedGaussianState.from_polarizations(np.array([[1.0, 0.5], [-1.0, 1.0]]))
        np.random.seed(TEST_SEED)
        samples = state.sample(11)
        assert samples.shape == (11, 2, 2)
        np.testing.assert_array_equal(samples[:, 0, 0], 0)
        np.testing.assert_array_equal(samples[:, 1, :], [[1, 0]] * 11)

    def test_evolve_matches_ensemble_and_dense(self, reduced_case):
        state, _, follow_up, evolved_density = reduced_case
        evolved = state.evolve(follow_up)
        assert evolved.wires == state.wires
        assert evolved.source_wires == state.source_wires
        np.testing.assert_allclose(
            evolved.covariance_matrix,
            self.dense_covariance(evolved_density, 3),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_evolve_with_torch_circuit_keeps_gradient(self):
        state = MixedGaussianState.from_polarizations(np.array([0.5, -0.2]))
        angles = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        evolved = state.evolve([CompRxRx(angles, wires=[0, 1])])
        value = evolved.expval(qml.PauliZ(0))
        assert isinstance(value, torch.Tensor) and value.requires_grad
        (gradient,) = torch.autograd.grad(value, angles)
        assert gradient.shape == (2,)

    def test_reduced_state_and_partial_trace_of_state(self, reduced_case):
        state, reduced_density, follow_up, evolved_density = reduced_case
        chained = state.evolve(follow_up).partial_trace([0])
        assert chained.wires.tolist() == [0, 1]
        assert chained.source_wires.tolist() == [2, 3]
        expected = np.asarray(qml.math.partial_trace(evolved_density, [0]))
        np.testing.assert_allclose(
            chained.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        kept = state.reduced_state([2, 1], reduced_wires=[5, 6])
        assert kept.wires.tolist() == [5, 6]
        assert kept.source_wires.tolist() == [2, 3]
        np.testing.assert_allclose(
            kept.covariance_matrix,
            state.covariance_matrix[2:, 2:],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_reduced_state_rejects_bad_wires(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2])
        with pytest.raises(ValueError, match="not wires of this state"):
            state.reduced_state([3])
        with pytest.raises(ValueError, match="not wires of this state"):
            state.partial_trace([3])
        with pytest.raises(ValueError, match="At least one wire"):
            state.reduced_state([])

    def test_init_rejects_shots_in_device_kwargs(self):
        with pytest.raises(ValueError, match="shots"):
            MixedGaussianState(np.zeros((4, 4)), device_kwargs={"shots": 5})

    def test_execute_rejects_shots_in_device_kwargs(self, reduced_case):
        state, _, follow_up, _ = reduced_case
        with pytest.raises(ValueError, match="shots"):
            state.execute(follow_up, output_type="probs", device_kwargs={"shots": 5})

    def test_atol_is_read_only(self):
        state = MixedGaussianState.from_polarizations([0.5, 0.2], atol=1e-6)
        assert state.atol == 1e-6
        with pytest.raises(AttributeError):
            state.atol = 1e-3  # type: ignore[misc]

    def test_single_wire_state_has_readouts_but_no_device(self):
        state = MixedGaussianState.from_polarizations([0.5])
        np.testing.assert_allclose(state.probs(), [0.75, 0.25], atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(state.expval(qml.PauliZ(0)), 0.5, atol=ATOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(state.purity, (1 + 0.25) / 2, atol=ATOL_SCALAR_COMPARISON)
        for call in (lambda: state.sample(3), lambda: state.execute([], output_type="probs"), lambda: state.evolve([])):
            with pytest.raises(ValueError, match="at least two wires"):
                call()

    def test_sample_on_shifted_wires(self):
        state = MixedGaussianState.from_polarizations([1.0, -1.0], wires=[3, 4])
        np.random.seed(TEST_SEED)
        samples = state.sample(5)
        np.testing.assert_array_equal(samples, [[0, 1]] * 5)
        probabilities = state.execute([], output_type="probs", mode="direct", shots=5)
        np.testing.assert_allclose(probabilities, [0.0, 1.0, 0.0, 0.0], atol=ATOL_MATRIX_COMPARISON)

    def test_evolve_and_direct_execute_carry_device_kwargs(self, reduced_case):
        state, _, follow_up, _ = reduced_case
        evolved = state.evolve(follow_up, pfaffian_chunk_size=4)
        assert evolved.device_kwargs == {"pfaffian_chunk_size": 4}
        np.random.seed(TEST_SEED)
        samples = state.execute(
            follow_up,
            output_type="samples",
            mode="direct",
            shots=3,
            device_kwargs={"sampling_strategy": "QubitByQubitSampling", "pfaffian_chunk_size": 4},
        )
        assert samples.shape == (3, 3)
        assert state.sample(2, sampling_strategy="QubitByQubitSampling").shape == (2, 3)

    def test_execute_parallel_batched_state(self):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        batched_angles = rng.normal(size=(3, 2))
        shared = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([CompRxRx(batched_angles, wires=[0, 1])] + shared)
        state = self.reduced_from_device(device, [3])
        assert state.batch_shape == (3,)
        follow_up = self.brick_wall(range(3), 1, rng)
        observable = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliZ(2)
        values = {
            mode: np.asarray(state.execute(follow_up, observable, "expval", mode=mode))
            for mode in ("parallel", "sequential", "direct")
        }
        probabilities = np.asarray(state.execute(follow_up, output_type="probs", mode="parallel"))
        np.random.seed(TEST_SEED)
        samples = state.execute(follow_up, output_type="samples", mode="sequential", shots=5)
        assert samples.shape == (5, 3, 3)
        unitary = self.dense_unitary(follow_up, 3)
        for batch_index in range(3):
            operations = [CompRxRx(batched_angles[batch_index], wires=[0, 1])] + shared
            psi = self.dense_state(np.zeros(n_wires, dtype=int), operations, n_wires)
            reduced = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [3]))
            evolved = unitary @ reduced @ unitary.conj().T
            expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(3))).real
            for mode, value in values.items():
                np.testing.assert_allclose(
                    value[batch_index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
                )
            np.testing.assert_allclose(probabilities[batch_index], np.diag(evolved).real, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["sequential", "direct"])
    def test_execute_batched_follow_up_circuit(self, reduced_case, mode):
        state, reduced_density, _, _ = reduced_case
        batched_angles = np.random.default_rng(TEST_SEED).normal(size=(2, 2))
        follow_up = [CompRyRy(batched_angles, wires=[1, 2])]
        observable = qml.PauliZ(1) @ qml.PauliZ(2)
        values = np.asarray(state.execute(follow_up, observable, "expval", mode=mode))
        assert values.shape == (2,)
        for batch_index in range(2):
            unitary = self.dense_unitary([CompRyRy(batched_angles[batch_index], wires=[1, 2])], 3)
            evolved = unitary @ reduced_density @ unitary.conj().T
            expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(3))).real
            np.testing.assert_allclose(
                values[batch_index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_gradient_through_partial_trace_matches_dense(self, mode):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        angles = torch.tensor(rng.normal(size=(3, 2)), dtype=torch.float64, requires_grad=True)
        basis_state = np.array([1, 0, 0, 1])

        def build(parameters):
            return [
                CompRxRx(parameters[0], wires=[0, 1]),
                CompRyRy(parameters[1], wires=[1, 2]),
                CompRxRx(parameters[2], wires=[2, 3]),
                fSWAP(wires=[1, 2]),
            ]

        follow_up = self.brick_wall(range(2), 1, rng)
        observable = qml.PauliX(0) @ qml.PauliX(1) + 0.4 * qml.PauliZ(1)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([qml.BasisState(basis_state, wires=range(n_wires))] + build(angles))
        value = self.reduced_from_device(device, [0, 3]).execute(follow_up, observable, "expval", mode=mode)
        (gradient,) = torch.autograd.grad(value, angles)

        qubit_device = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(qubit_device, interface="torch")
        def dense_circuit(parameters):
            qml.BasisState(basis_state, wires=range(n_wires))
            with qml.QueuingManager.stop_recording():
                operations = build(parameters)
            for operation in operations:
                qml.apply(operation)
            return qml.state()

        reference_angles = angles.detach().clone().requires_grad_()
        psi = dense_circuit(reference_angles)
        reduced = qml.math.partial_trace(torch.outer(psi, psi.conj()), [0, 3])
        unitary = torch.tensor(self.dense_unitary(follow_up, 2))
        evolved = unitary @ reduced @ unitary.conj().T
        reference = torch.trace(evolved @ torch.tensor(qml.matrix(observable, wire_order=range(2)))).real
        (reference_gradient,) = torch.autograd.grad(reference, reference_angles)
        torch.testing.assert_close(
            value.detach(), reference.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            gradient, reference_gradient, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_init_rejects_displacement_of_wrong_shape(self):
        with pytest.raises(ValueError, match="displacement must have shape"):
            MixedGaussianState(np.zeros((4, 4)), displacement=np.zeros(3))
        with pytest.raises(ValueError, match="displacement must have shape"):
            MixedGaussianState(np.zeros((2, 4, 4)), displacement=np.zeros(4))

    def test_zero_displacement_is_not_displaced(self):
        covariance = block_diagonal_covariance(np.array([0.5, -0.2]))
        state = MixedGaussianState(covariance, displacement=np.zeros(4))
        reference = MixedGaussianState(covariance)
        assert not state.is_displaced
        assert not reference.is_displaced
        assert state.parity_wire is None
        assert state.ensemble_wires == state.wires
        assert state.sptm.shape == (4, 4)
        assert reference.displacement.shape == (4,)
        np.testing.assert_allclose(reference.displacement, 0.0, atol=ATOL_MATRIX_COMPARISON)
        observable = qml.PauliX(0)
        assert state.ensemble_observable(observable) is observable
        assert "is_displaced" not in repr(state)

    def test_displaced_state_properties(self, displaced_case):
        state, _, _, _ = displaced_case
        assert state.is_displaced
        assert state.parity_wire == -1
        assert state.ensemble_wires.tolist() == [-1, 0, 1, 2]
        assert repr(state).endswith("is_displaced=True)")
        extended = np.asarray(state.extended_covariance_matrix)
        assert extended.shape == (7, 7)
        np.testing.assert_allclose(extended[:6, :6], state.covariance_matrix, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(extended[:6, 6], state.displacement, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(extended[6, :6], -np.asarray(state.displacement), atol=ATOL_MATRIX_COMPARISON)
        sptm = np.asarray(state.sptm)
        assert sptm.shape == (8, 8)
        np.testing.assert_allclose(sptm @ sptm.T, np.eye(8), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(np.linalg.det(sptm), 1.0, atol=ATOL_SCALAR_COMPARISON)
        assert state.sptm_operation.wires.tolist() == [-1, 0, 1, 2]
        np.testing.assert_allclose(np.sum(np.asarray(state.weights)), 1.0, atol=ATOL_SCALAR_COMPARISON)

    def test_displaced_state_parity_wire_follows_the_first_wire(self):
        covariance = block_diagonal_covariance(np.array([0.0, 1.0]))
        state = MixedGaussianState(covariance, wires=[3, 4], displacement=np.array([0.6, 0.0, 0.0, 0.0]))
        assert state.parity_wire == 2
        assert state.ensemble_wires.tolist() == [2, 3, 4]
        prep, sptm_operation = state.pure_state_operations(0)
        assert prep.wires.tolist() == [2, 3, 4]
        assert sptm_operation.wires.tolist() == [2, 3, 4]

    def test_displaced_density_matrix_matches_qubit_partial_trace(self, displaced_case):
        state, reduced_density, _, _ = displaced_case
        np.testing.assert_allclose(
            state.density_matrix(), reduced_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_displaced_entropy_and_purity_match_dense(self, displaced_case):
        state, reduced_density, _, _ = displaced_case
        eigenvalues = np.linalg.eigvalsh(reduced_density)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        np.testing.assert_allclose(
            state.entropy(),
            -np.sum(eigenvalues * np.log(eigenvalues)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            state.purity,
            np.trace(reduced_density @ reduced_density).real,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        assert not state.is_pure

    def test_displaced_pure_product_state_is_pure(self):
        amplitudes = self.random_product_amplitudes(3, np.random.default_rng(TEST_SEED))
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([ProductState(amplitudes, wires=range(3))])
        state = self.displaced_from_device(device, [0, 1, 2])
        assert state.is_displaced
        assert state.is_pure
        assert state.num_pure_states == 1
        np.testing.assert_allclose(state.entropy(), 0.0, atol=ATOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(state.purity, 1.0, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("pauli_word", ["XII", "IYI", "ZIX", "XYX", "XXI", "ZZI", "YZX", "IZI"])
    def test_displaced_pure_states_reproduce_expectation_values(self, displaced_case, pauli_word):
        state, reduced_density, _, _ = displaced_case
        observable = qml.pauli.string_to_pauli_word(pauli_word)
        expected = np.trace(reduced_density @ qml.matrix(observable, wire_order=range(3))).real
        device = NonInteractingFermionicDevice(wires=state.ensemble_wires)
        total = 0.0
        for weight, prep, sptm_operation in state.pure_states():
            value = device.execute_generator(
                [prep, sptm_operation],
                observable=state.ensemble_observable(observable),
                output_type="expval",
                reset=True,
            )
            total += float(weight) * float(np.asarray(value))
        np.testing.assert_allclose(total, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        np.testing.assert_allclose(
            state.expval(observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_displaced_ensemble_observable_dresses_the_odd_words(self, displaced_case):
        state, _, _, _ = displaced_case
        even_word = qml.PauliX(0) @ qml.PauliY(2)
        assert state.ensemble_observable(even_word) is even_word
        odd_word = state.ensemble_observable(qml.PauliZ(0) @ qml.PauliY(1))
        assert qml.pauli.pauli_word_to_string(odd_word, wire_map={-1: 0, 0: 1, 1: 2, 2: 3}) == "XZYI"
        projector = qml.Projector(np.array([1, 0]), wires=[0, 2])
        assert state.ensemble_observable(projector) is projector
        hamiltonian = state.ensemble_observable(0.5 * qml.PauliX(1) + 2.0 * qml.PauliZ(2))
        coefficients, words = hamiltonian.terms()
        np.testing.assert_allclose(coefficients, [0.5, 2.0], atol=ATOL_SCALAR_COMPARISON)
        assert [qml.pauli.pauli_word_to_string(word, wire_map={-1: 0, 0: 1, 1: 2, 2: 3}) for word in words] == [
            "XIXI",
            "IIIZ",
        ]
        batch = state.ensemble_observable(BatchHamiltonian([1.0, 3.0], [qml.PauliY(0), qml.PauliZ(1)]))
        assert isinstance(batch, BatchHamiltonian)
        assert -1 in batch.ops[0].wires.tolist()
        assert -1 not in batch.ops[1].wires.tolist()

    def test_displaced_ensemble_observable_rejects_non_pauli_observable(self, displaced_case):
        state, _, follow_up, _ = displaced_case
        hermitian = qml.Hermitian(np.eye(2), wires=[0])
        with pytest.raises(NotImplementedError, match="pure-state decomposition of a displaced"):
            state.ensemble_observable(hermitian)
        with pytest.raises(NotImplementedError, match="pure-state decomposition of a displaced"):
            state.execute(follow_up, hermitian, "expval", mode="parallel")

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_displaced_execute_expval_matches_dense(self, displaced_case, mode):
        state, _, follow_up, evolved_density = displaced_case
        observables = [
            qml.PauliX(0),
            qml.PauliY(2),
            qml.PauliZ(0) @ qml.PauliX(1),
            qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2),
            qml.PauliX(0) @ qml.PauliY(1) + 0.5 * qml.PauliZ(2) + 0.3 * qml.PauliX(1) - 1.1 * qml.PauliY(0),
        ]
        for observable in observables:
            expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
            value = state.execute(follow_up, observable, "expval", mode=mode)
            np.testing.assert_allclose(value, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_displaced_execute_batch_hamiltonian_and_projector_match_dense(self, displaced_case, mode):
        state, _, follow_up, evolved_density = displaced_case
        terms = [qml.PauliY(0), qml.PauliX(1) @ qml.PauliX(2), qml.PauliZ(0) @ qml.PauliX(2)]
        coefficients = [0.2, -0.7, 1.3]
        expected = [
            coefficient * np.trace(evolved_density @ qml.matrix(term, wire_order=range(3))).real
            for coefficient, term in zip(coefficients, terms)
        ]
        value = state.execute(follow_up, BatchHamiltonian(coefficients, terms), "expval", mode=mode)
        np.testing.assert_allclose(value, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        projector = qml.Projector(np.array([1, 0]), wires=[0, 2])
        marginal = np.asarray(qml.math.partial_trace(evolved_density, [1]))
        np.testing.assert_allclose(
            state.execute(follow_up, projector, "expval", mode=mode),
            np.diag(marginal).real[2],
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_displaced_execute_probs_and_samples_match_dense(self, displaced_case, mode):
        state, _, follow_up, evolved_density = displaced_case
        probabilities = state.execute(follow_up, output_type="probs", mode=mode)
        np.testing.assert_allclose(
            probabilities, np.diag(evolved_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        marginal = np.asarray(qml.math.partial_trace(evolved_density, [1]))
        np.testing.assert_allclose(
            state.execute(follow_up, output_type="probs", mode=mode, wires=[0, 2]),
            np.diag(marginal).real,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.random.seed(TEST_SEED)
        samples = state.execute(follow_up, output_type="samples", mode=mode, shots=N_SHOTS)
        assert samples.shape == (N_SHOTS, 3)
        frequencies = np.bincount(samples @ (2 ** np.arange(3)[::-1]), minlength=8) / N_SHOTS
        np.testing.assert_allclose(frequencies, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)
        estimated = state.execute(follow_up, output_type="probs", mode=mode, shots=N_SHOTS)
        np.testing.assert_allclose(estimated, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)

    def test_displaced_evolve_and_chained_partial_trace_match_dense(self, displaced_case):
        state, _, follow_up, evolved_density = displaced_case
        evolved = state.evolve(follow_up)
        assert evolved.is_displaced
        np.testing.assert_allclose(
            evolved.density_matrix(), evolved_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            np.linalg.norm(evolved.displacement),
            np.linalg.norm(state.displacement),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chained = evolved.partial_trace([2])
        assert chained.wires.tolist() == [0, 1]
        assert chained.source_wires.tolist() == [0, 1]
        np.testing.assert_allclose(
            chained.density_matrix(),
            np.asarray(qml.math.partial_trace(evolved_density, [2])),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_displaced_reduced_state_warns_when_front_wires_are_traced(self, displaced_case):
        state, reduced_density, _, _ = displaced_case
        with pytest.warns(UserWarning, match="fermionic partial trace"):
            reduced = state.partial_trace([0])
        assert reduced.source_wires.tolist() == [1, 2]
        qubit_reduced = np.asarray(qml.math.partial_trace(reduced_density, [0]))
        np.testing.assert_allclose(
            reduced.probs(), np.diag(qubit_reduced).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        even_word = qml.PauliX(0) @ qml.PauliY(1)
        np.testing.assert_allclose(
            reduced.expval(even_word),
            np.trace(qubit_reduced @ qml.matrix(even_word, wire_order=range(2))).real,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        # An odd word of the reduced state carries the Z of the traced-out wire in front of it.
        dressed = qml.PauliZ(0) @ qml.PauliX(1)
        np.testing.assert_allclose(
            reduced.expval(qml.PauliX(0)),
            np.trace(reduced_density @ qml.matrix(dressed, wire_order=range(3))).real,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_displaced_single_wire_state(self):
        amplitudes = np.array([[np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.4j)], [np.sqrt(0.6), np.sqrt(0.4)]])
        operations = [CompRxRx(np.array([0.3, -0.8]), wires=[0, 1])]
        device = NonInteractingFermionicDevice(wires=2)
        device.apply([ProductState(amplitudes, wires=[0, 1])] + operations)
        state = self.displaced_from_device(device, [0])
        psi = self.dense_product_state(amplitudes, operations, 2)
        reduced_density = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [1]))
        np.testing.assert_allclose(
            state.density_matrix(), reduced_density, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        for observable in (qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)):
            expected = np.trace(reduced_density @ qml.matrix(observable)).real
            np.testing.assert_allclose(
                state.expval(observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            np.testing.assert_allclose(
                state.execute([], observable, "expval", mode="parallel"),
                expected,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
        with pytest.raises(ValueError, match="at least two wires"):
            state.evolve([])

    def test_displaced_batched_state_matches_dense_in_all_modes(self):
        rng = np.random.default_rng(TEST_SEED + 3)
        n_wires = 4
        amplitudes = np.stack([self.random_product_amplitudes(n_wires, rng) for _ in range(3)])
        operations = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires))] + operations)
        state = self.displaced_from_device(device, [0, 1])
        assert state.batch_shape == (3,)
        follow_up = self.brick_wall(range(2), 1, rng)
        unitary = self.dense_unitary(follow_up, 2)
        observable = qml.PauliX(0) @ qml.PauliZ(1) + 0.6 * qml.PauliY(1) - 0.2 * qml.PauliX(0) @ qml.PauliX(1)
        values = {
            mode: np.asarray(state.execute(follow_up, observable, "expval", mode=mode))
            for mode in ("parallel", "sequential", "direct")
        }
        probabilities = np.asarray(state.execute(follow_up, output_type="probs", mode="parallel"))
        for batch_index in range(3):
            psi = self.dense_product_state(amplitudes[batch_index], operations, n_wires)
            reduced = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [2, 3]))
            np.testing.assert_allclose(
                np.asarray(state.density_matrix())[batch_index],
                reduced,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
            evolved = unitary @ reduced @ unitary.conj().T
            expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(2))).real
            for value in values.values():
                np.testing.assert_allclose(
                    value[batch_index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
                )
            np.testing.assert_allclose(probabilities[batch_index], np.diag(evolved).real, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_displaced_gradient_through_partial_trace_matches_dense(self, mode):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        angles = torch.tensor(rng.normal(size=(3, 2)), dtype=torch.float64, requires_grad=True)
        tilts = torch.tensor(rng.uniform(0.3, 2.5, size=n_wires), dtype=torch.float64, requires_grad=True)
        phases = torch.tensor(rng.normal(size=n_wires), dtype=torch.complex128)

        def amplitudes_of(tilt_angles):
            return torch.stack(
                [torch.cos(tilt_angles / 2) + 0j, torch.sin(tilt_angles / 2) * torch.exp(1j * phases)], dim=-1
            )

        def build(parameters):
            return [
                CompRxRx(parameters[0], wires=[0, 1]),
                CompRyRy(parameters[1], wires=[1, 2]),
                CompRxRx(parameters[2], wires=[2, 3]),
                fSWAP(wires=[1, 2]),
            ]

        follow_up = self.brick_wall(range(2), 1, rng)
        observable = qml.PauliX(0) + 0.7 * qml.PauliY(0) @ qml.PauliZ(1) + 0.4 * qml.PauliX(0) @ qml.PauliX(1)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes_of(tilts), wires=range(n_wires))] + build(angles))
        value = self.displaced_from_device(device, [0, 1]).execute(follow_up, observable, "expval", mode=mode)
        gradient_angles, gradient_tilts = torch.autograd.grad(value, (angles, tilts))

        qubit_device = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(qubit_device, interface="torch")
        def dense_circuit(parameters, initial_state):
            qml.StatePrep(initial_state, wires=range(n_wires))
            with qml.QueuingManager.stop_recording():
                operations = build(parameters)
            for operation in operations:
                qml.apply(operation)
            return qml.state()

        reference_angles = angles.detach().clone().requires_grad_()
        reference_tilts = tilts.detach().clone().requires_grad_()
        reference_amplitudes = amplitudes_of(reference_tilts)
        initial_state = reference_amplitudes[0]
        for wire in range(1, n_wires):
            initial_state = torch.kron(initial_state, reference_amplitudes[wire])
        psi = dense_circuit(reference_angles, initial_state)
        reduced = qml.math.partial_trace(torch.outer(psi, psi.conj()), [2, 3])
        unitary = torch.tensor(self.dense_unitary(follow_up, 2))
        evolved = unitary @ reduced @ unitary.conj().T
        reference = torch.trace(evolved @ torch.tensor(qml.matrix(observable, wire_order=range(2)))).real
        reference_gradients = torch.autograd.grad(reference, (reference_angles, reference_tilts))
        torch.testing.assert_close(
            value.detach(), reference.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            gradient_angles, reference_gradients[0], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            gradient_tilts, reference_gradients[1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    @pytest.mark.parametrize("bits", [[0], [1], [1, 0]])
    def test_insert_basis_ancilla_into_displaced_state_matches_dense(self, displaced_case, index, bits):
        state, reduced_density, _, _ = displaced_case
        joint = state.insert(index, bits)
        assert joint.num_wires == 3 + len(bits)
        assert joint.wires.tolist() == list(range(3 + len(bits)))
        expected = self.dense_insert(reduced_density, self.dense_ancilla(np.eye(2)[bits]), index)
        np.testing.assert_allclose(
            joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    @pytest.mark.parametrize("bits", [[0], [1], [0, 1, 1]])
    def test_insert_basis_ancilla_into_state_without_displacement_matches_dense(self, reduced_case, index, bits):
        state, reduced_density, _, _ = reduced_case
        joint = state.insert(index, bits)
        assert not joint.is_displaced
        expected = self.dense_insert(reduced_density, self.dense_ancilla(np.eye(2)[bits]), index)
        np.testing.assert_allclose(
            joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("n_ancillas", [1, 2])
    @pytest.mark.parametrize("case_name", ["displaced_case", "reduced_case"])
    def test_insert_product_ancilla_in_front_matches_dense(self, request, case_name, n_ancillas):
        state, reduced_density, _, _ = request.getfixturevalue(case_name)
        amplitudes = self.random_product_amplitudes(n_ancillas, np.random.default_rng(TEST_SEED + n_ancillas))
        joint = state.insert(0, amplitudes)
        assert joint.is_displaced
        expected = self.dense_insert(reduced_density, self.dense_ancilla(amplitudes), 0)
        np.testing.assert_allclose(
            joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_insert_product_ancilla_anywhere_in_a_product_state_matches_dense(self, index):
        rng = np.random.default_rng(TEST_SEED + 20)
        amplitudes = self.random_product_amplitudes(3, rng)
        ancilla = self.random_product_amplitudes(2, rng)
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([ProductState(amplitudes, wires=range(3))])
        state = self.displaced_from_device(device, [0, 1, 2])
        joint = state.insert(index, ancilla)
        expected = self.dense_insert(self.dense_ancilla(amplitudes), self.dense_ancilla(ancilla), index)
        np.testing.assert_allclose(
            joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert joint.is_pure

    def test_insert_product_ancilla_behind_a_pure_block_matches_dense(self):
        # Wires 0-1 are in a pure entangled state and wires 2-3 in a mixed state that is not correlated with
        # them: an ancilla with superposed qubits fits in front, and right behind the pure block.
        rng = np.random.default_rng(TEST_SEED + 21)
        n_wires = 6
        amplitudes = self.random_product_amplitudes(n_wires, rng)
        operations = self.brick_wall([0, 1], 2, rng) + self.brick_wall([2, 3, 4, 5], 2, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires))] + operations)
        state = self.displaced_from_device(device, [0, 1, 2, 3])
        psi = self.dense_product_state(amplitudes, operations, n_wires)
        reduced_density = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [4, 5]))
        ancilla = self.random_product_amplitudes(1, rng)
        for index in (0, 2):
            expected = self.dense_insert(reduced_density, self.dense_ancilla(ancilla), index)
            np.testing.assert_allclose(
                state.insert(index, ancilla).density_matrix(),
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
        for index in (1, 3, 4):
            with pytest.raises(ValueError, match="pure state"):
                state.insert(index, ancilla)

    def test_append_product_ancilla_to_a_pure_entangled_state_matches_dense(self):
        rng = np.random.default_rng(TEST_SEED + 22)
        amplitudes = self.random_product_amplitudes(3, rng)
        operations = self.brick_wall(range(3), 2, rng)
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([ProductState(amplitudes, wires=range(3))] + operations)
        state = self.displaced_from_device(device, [0, 1, 2])
        psi = self.dense_product_state(amplitudes, operations, 3)
        ancilla = self.random_product_amplitudes(2, rng)
        joint = state.append(ancilla)
        expected = self.dense_insert(np.outer(psi, psi.conj()), self.dense_ancilla(ancilla), 3)
        np.testing.assert_allclose(
            joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_insert_product_ancilla_behind_mixed_wires_is_rejected(self, displaced_case, index):
        state, _, _, _ = displaced_case
        ancilla = self.random_product_amplitudes(1, np.random.default_rng(TEST_SEED))
        with pytest.raises(ValueError, match="pure state"):
            state.insert(index, ancilla)

    def test_append_is_insert_at_the_end(self, displaced_case):
        state, _, _, _ = displaced_case
        appended = state.append([1, 0])
        inserted = state.insert(3, [1, 0])
        assert appended.wires.tolist() == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(appended.covariance_matrix, inserted.covariance_matrix, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(appended.displacement, inserted.displacement, atol=ATOL_MATRIX_COMPARISON)

    def test_insert_counts_a_negative_index_from_the_end(self, displaced_case):
        state, _, _, _ = displaced_case
        np.testing.assert_allclose(
            state.insert(-1, [1]).covariance_matrix,
            state.insert(2, [1]).covariance_matrix,
            atol=ATOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            state.insert(-3, [1]).covariance_matrix,
            state.insert(0, [1]).covariance_matrix,
            atol=ATOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("index", [4, -4, 10])
    def test_insert_rejects_an_index_out_of_range(self, displaced_case, index):
        state, _, _, _ = displaced_case
        with pytest.raises(ValueError, match="index"):
            state.insert(index, [0])

    def test_insert_accepts_the_forms_of_a_basis_ancilla(self, displaced_case):
        state, _, _, _ = displaced_case
        reference = state.insert(1, [1, 0])
        forms = [
            np.array([1, 0]),
            qml.BasisState(np.array([1, 0]), wires=[7, 8]),
            ProductState.from_basis_state(np.array([1, 0]), wires=["a", "b"]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ]
        for ancilla in forms:
            joint = state.insert(1, ancilla)
            np.testing.assert_allclose(
                joint.covariance_matrix, reference.covariance_matrix, atol=ATOL_MATRIX_COMPARISON
            )
            np.testing.assert_allclose(joint.displacement, reference.displacement, atol=ATOL_MATRIX_COMPARISON)
        single = state.insert(1, 1)
        np.testing.assert_allclose(
            single.covariance_matrix, state.insert(1, [1]).covariance_matrix, atol=ATOL_MATRIX_COMPARISON
        )

    def test_insert_accepts_a_product_state_operation(self, displaced_case):
        state, _, _, _ = displaced_case
        amplitudes = self.random_product_amplitudes(2, np.random.default_rng(TEST_SEED))
        from_array = state.insert(0, amplitudes)
        from_operation = state.insert(0, ProductState(amplitudes, wires=[5, 6]))
        np.testing.assert_allclose(
            from_operation.covariance_matrix, from_array.covariance_matrix, atol=ATOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(from_operation.displacement, from_array.displacement, atol=ATOL_MATRIX_COMPARISON)

    def test_insert_rejects_bits_that_are_not_zero_or_one(self, displaced_case):
        state, _, _, _ = displaced_case
        with pytest.raises(ValueError, match="bits"):
            state.insert(0, [0.6, 0.8])
        with pytest.raises(ValueError, match="bits"):
            state.insert(0, [2])

    def test_insert_wires_and_settings_of_the_new_state(self):
        covariance = block_diagonal_covariance(np.array([0.5, -1.0]))
        state = MixedGaussianState(
            covariance, wires=[3, 4], atol=1e-6, device_kwargs={"pfaffian_chunk_size": 4}, source_wires=[8, 9]
        )
        joint = state.insert(1, [1])
        assert joint.wires.tolist() == [3, 4, 5]
        assert joint.source_wires.tolist() == [3, 4, 5]
        assert joint.atol == 1e-6
        assert joint.device_kwargs == {"pfaffian_chunk_size": 4}
        np.testing.assert_allclose(
            np.sort(np.abs(np.asarray(joint.mode_polarizations))), [0.5, 1.0, 1.0], atol=ATOL_MATRIX_COMPARISON
        )
        relabeled = state.insert(1, [1], wires=[0, 1, 2])
        assert relabeled.wires.tolist() == [0, 1, 2]
        with pytest.raises(ValueError, match="wires"):
            state.insert(1, [1], wires=[0, 1])

    def test_insert_keeps_the_backend_of_the_state(self):
        polarizations = np.array([0.4, -0.7])
        system_density = np.diag([0.7 * 0.15, 0.7 * 0.85, 0.3 * 0.15, 0.3 * 0.85]).astype(complex)
        ancilla = self.random_product_amplitudes(1, np.random.default_rng(TEST_SEED))
        expected = self.dense_insert(system_density, self.dense_ancilla(ancilla), 0)
        numpy_joint = MixedGaussianState.from_polarizations(polarizations).insert(0, ancilla)
        assert isinstance(numpy_joint.covariance_matrix, np.ndarray)
        assert isinstance(numpy_joint.displacement, np.ndarray)
        np.testing.assert_allclose(
            numpy_joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch_joint = MixedGaussianState.from_polarizations(torch.tensor(polarizations)).insert(0, ancilla)
        assert isinstance(torch_joint.covariance_matrix, torch.Tensor)
        assert isinstance(torch_joint.displacement, torch.Tensor)
        np.testing.assert_allclose(
            torch_joint.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_insert_into_a_batched_state_matches_dense(self):
        rng = np.random.default_rng(TEST_SEED + 23)
        n_wires = 4
        amplitudes = np.stack([self.random_product_amplitudes(n_wires, rng) for _ in range(3)])
        operations = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires))] + operations)
        state = self.displaced_from_device(device, [0, 1])
        ancilla = self.random_product_amplitudes(1, rng)
        in_front = state.insert(0, ancilla)
        in_between = state.insert(1, [1])
        assert in_front.batch_shape == (3,)
        assert in_between.batch_shape == (3,)
        for batch_index in range(3):
            psi = self.dense_product_state(amplitudes[batch_index], operations, n_wires)
            reduced = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [2, 3]))
            np.testing.assert_allclose(
                np.asarray(in_front.density_matrix())[batch_index],
                self.dense_insert(reduced, self.dense_ancilla(ancilla), 0),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
            np.testing.assert_allclose(
                np.asarray(in_between.density_matrix())[batch_index],
                self.dense_insert(reduced, self.dense_ancilla(np.eye(2)[[1]]), 1),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    def test_insert_a_batch_of_ancillas_matches_dense(self, displaced_case):
        state, reduced_density, _, _ = displaced_case
        rng = np.random.default_rng(TEST_SEED + 24)
        ancillas = np.stack([self.random_product_amplitudes(2, rng) for _ in range(3)])
        joint = state.insert(0, ancillas)
        assert joint.batch_shape == (3,)
        for batch_index in range(3):
            np.testing.assert_allclose(
                np.asarray(joint.density_matrix())[batch_index],
                self.dense_insert(reduced_density, self.dense_ancilla(ancillas[batch_index]), 0),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    def test_append_then_partial_trace_returns_the_state(self, displaced_case):
        state, _, _, _ = displaced_case
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            recovered = state.append([1, 0]).partial_trace([3, 4])
        np.testing.assert_allclose(recovered.covariance_matrix, state.covariance_matrix, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(recovered.displacement, state.displacement, atol=ATOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("mode", ["parallel", "sequential", "direct"])
    def test_circuit_on_a_state_with_an_inserted_ancilla_matches_dense(self, displaced_case, mode):
        # One more round with a fresh ancilla: insert it, couple it to the system, and trace it out again.
        state, reduced_density, _, _ = displaced_case
        rng = np.random.default_rng(TEST_SEED + 25)
        ancilla = self.random_product_amplitudes(1, rng)
        joint = state.insert(0, ancilla)
        operations = self.brick_wall(range(4), 1, rng)
        unitary = self.dense_unitary(operations, 4)
        joint_density = self.dense_insert(reduced_density, self.dense_ancilla(ancilla), 0)
        evolved_density = unitary @ joint_density @ unitary.conj().T
        observable = qml.PauliX(0) + 0.5 * qml.PauliZ(1) @ qml.PauliY(2) - 0.3 * qml.PauliX(2) @ qml.PauliX(3)
        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(4))).real
        np.testing.assert_allclose(
            joint.execute(operations, observable, "expval", mode=mode),
            expected,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            joint.execute(operations, output_type="probs", mode=mode),
            np.diag(evolved_density).real,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_gradient_through_an_inserted_ancilla_matches_dense(self):
        rng = np.random.default_rng(TEST_SEED + 26)
        covariance = torch.tensor(
            np.asarray(MixedGaussianState.from_polarizations(np.array([0.4, -0.7])).covariance_matrix),
            dtype=torch.float64,
        )
        system_density = torch.tensor(np.diag([0.7 * 0.15, 0.7 * 0.85, 0.3 * 0.15, 0.3 * 0.85]), dtype=torch.complex128)
        tilt = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
        angles = torch.tensor(rng.normal(size=(2, 2)), dtype=torch.float64, requires_grad=True)
        observable = qml.PauliX(0) @ qml.PauliZ(1) + 0.6 * qml.PauliY(1) @ qml.PauliX(2) - 0.8 * qml.PauliZ(2)

        def ancilla_of(tilt_angle):
            return torch.stack([torch.cos(tilt_angle / 2) + 0j, torch.sin(tilt_angle / 2) * np.exp(0.3j)])[None, :]

        def build(parameters):
            return [CompRxRx(parameters[0], wires=[0, 1]), CompRyRy(parameters[1], wires=[1, 2])]

        state = MixedGaussianState(covariance).insert(0, ancilla_of(tilt))
        value = state.execute(build(angles), observable, "expval", mode="direct")
        gradient_tilt, gradient_angles = torch.autograd.grad(value, (tilt, angles))

        reference_tilt = tilt.detach().clone().requires_grad_()
        reference_angles = angles.detach().clone().requires_grad_()
        ancilla_vector = ancilla_of(reference_tilt)[0]
        joint_density = torch.kron(torch.outer(ancilla_vector, ancilla_vector.conj()), system_density)
        unitary = qml.matrix(qml.tape.QuantumScript(build(reference_angles)), wire_order=range(3))
        evolved = unitary @ joint_density @ unitary.conj().T
        reference = torch.trace(evolved @ torch.tensor(qml.matrix(observable, wire_order=range(3)))).real
        reference_gradients = torch.autograd.grad(reference, (reference_tilt, reference_angles))
        torch.testing.assert_close(
            value.detach(), reference.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            gradient_tilt, reference_gradients[0], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            gradient_angles, reference_gradients[1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
