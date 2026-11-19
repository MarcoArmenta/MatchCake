import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import MixedGaussianState, NonInteractingFermionicDevice
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.single_particle_transition_matrices import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake.operations.state_preparation import PlusState, ProductState
from matchcake.utils.jordan_wigner import JordanWigner
from matchcake.utils.majorana import get_majorana

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)


class TestNonInteractingFermionicDevicePartialTrace:
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
        if len(operations) == 0:
            return psi
        unitary = np.asarray(qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(n_wires)))
        return unitary @ psi

    @staticmethod
    def dressed_majorana_product(pauli_word: str, kept_wires, n_wires: int) -> np.ndarray:
        """Dense operator, in the full system, of a Pauli word of the reduced system.

        The Pauli word is mapped to a Majorana product with the Jordan-Wigner transformation of the reduced
        system, and every reduced Majorana is replaced by the Majorana of the kept wire it comes from.

        :param pauli_word: Pauli word on the reduced wires, e.g. ``"XZY"``.
        :type pauli_word: str
        :param kept_wires: Kept wires, in order.
        :type kept_wires: List[int]
        :param n_wires: Number of wires of the full system.
        :type n_wires: int
        :return: Dense matrix of shape ``(2^n, 2^n)``.
        :rtype: np.ndarray
        """
        kept_wires = list(kept_wires)
        indices, phase = JordanWigner(len(kept_wires)).pauli_to_majorana(pauli_word, list(range(len(kept_wires))))
        majorana_map = MixedGaussianState.majorana_indices(kept_wires)
        operator = np.eye(2**n_wires, dtype=complex)
        for index in indices:
            operator = operator @ get_majorana(int(majorana_map[int(index)]), n_wires)
        return phase * operator

    @staticmethod
    def run(state: MixedGaussianState, operations, observable=None, output_type: str = "expval", **kwargs):
        """Prepare a reduced state on a device with its wires, apply a circuit and return an output.

        :param state: The state to prepare.
        :type state: MixedGaussianState
        :param operations: Operations of the circuit, on the wires of the state.
        :type operations: List[Operation]
        :param observable: Observable for ``output_type="expval"``.
        :type observable: Optional[Operator]
        :param output_type: ``"expval"`` or ``"probs"``.
        :type output_type: str
        :param kwargs: Forwarded to ``execute_generator``.
        :return: The output of the device.
        :rtype: Any
        """
        device = NonInteractingFermionicDevice(wires=state.wires)
        return device.execute_generator(
            [state.state_prep()] + list(operations),
            observable=observable,
            output_type=output_type,
            reset=True,
            **kwargs,
        )

    @staticmethod
    def move_behind(traced_wires, n_wires: int):
        """fSWAP gates that move the traced wires behind the kept ones, both groups keeping their order.

        :param traced_wires: Wires to move to the end of the register.
        :type traced_wires: List[int]
        :param n_wires: Number of wires of the register.
        :type n_wires: int
        :return: The fSWAP operations.
        :rtype: List[Operation]
        """
        order = list(range(n_wires))
        swaps = []
        for wire in sorted(traced_wires, reverse=True):
            position = order.index(wire)
            while position < n_wires - 1 and order[position + 1] not in traced_wires:
                swaps.append(fSWAP(wires=[position, position + 1]))
                order[position], order[position + 1] = order[position + 1], order[position]
                position += 1
        return swaps

    @pytest.fixture
    def prepared_device(self):
        """A 5-wire device after a random circuit on a basis state, with the dense state vector."""
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 5
        basis_state = np.array([0, 1, 1, 0, 1])
        operations = self.brick_wall(range(n_wires), 2, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([qml.BasisState(basis_state, wires=range(n_wires))] + operations)
        psi = self.dense_state(basis_state, operations, n_wires)
        return device, psi, rng

    @pytest.fixture
    def prepared_product_device(self):
        """A 5-wire device after a random circuit on a random product state, with the dense state vector."""
        rng = np.random.default_rng(TEST_SEED + 1)
        n_wires = 5
        amplitudes = self.random_product_amplitudes(n_wires, rng)
        operations = self.brick_wall(range(n_wires), 2, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires))] + operations)
        psi = self.dense_product_state(amplitudes, operations, n_wires)
        return device, psi, rng

    def test_partial_trace_returns_a_mixed_gaussian_state(self, prepared_device):
        device, _, _ = prepared_device
        state = device.partial_trace([0, 1])
        assert isinstance(state, MixedGaussianState)
        assert state.wires.tolist() == [0, 1, 2]
        assert state.source_wires.tolist() == [2, 3, 4]
        assert state.atol == MixedGaussianState.DEFAULT_ATOL
        assert not state.is_displaced

    def test_reduced_state_options(self, prepared_device):
        device, _, _ = prepared_device
        state = device.reduced_state([3, 1], reduced_wires=[10, 11], atol=1e-4, fermionic=True)
        assert state.wires.tolist() == [10, 11]
        assert state.source_wires.tolist() == [1, 3]
        assert state.atol == 1e-4

    def test_reduced_state_covariance_is_principal_submatrix(self, prepared_device):
        device, _, _ = prepared_device
        state = device.reduced_state([1, 3], fermionic=True)
        indices = np.array([2, 3, 6, 7])
        expected = np.asarray(device.covariance_matrix)[np.ix_(indices, indices)]
        np.testing.assert_allclose(
            state.covariance_matrix, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("traced_wires", [[0], [4], [0, 4], [3, 4], [0, 1, 2]])
    def test_contiguous_partial_trace_matches_qubit_partial_trace(self, prepared_device, traced_wires):
        device, psi, _ = prepared_device
        state = device.partial_trace(traced_wires)
        expected = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), traced_wires))
        np.testing.assert_allclose(
            state.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_number_of_mixed_modes_is_bounded_by_traced_wires(self, prepared_device):
        device, _, _ = prepared_device
        assert device.partial_trace([2], fermionic=True).num_mixed_modes <= 1
        assert device.partial_trace([0, 1]).num_mixed_modes <= 2
        assert device.partial_trace([]).num_mixed_modes == 0
        assert device.partial_trace([]).is_pure

    @pytest.mark.parametrize("traced_wires", [[1], [1, 3], [0, 2]])
    def test_non_contiguous_partial_trace_must_be_requested_as_fermionic(self, prepared_device, traced_wires):
        device, _, _ = prepared_device
        with pytest.raises(ValueError, match="fermionic=True"):
            device.partial_trace(traced_wires)

    @pytest.mark.parametrize("traced_wires", [[1], [1, 3], [0, 2]])
    @pytest.mark.parametrize("pauli_word", ["XX", "ZZ", "YX", "XZ"])
    def test_non_contiguous_partial_trace_is_the_fermionic_one(self, prepared_device, traced_wires, pauli_word):
        device, psi, _ = prepared_device
        state = device.partial_trace(traced_wires, fermionic=True)
        kept_wires = state.source_wires.tolist()
        word = pauli_word + "I" * (len(kept_wires) - len(pauli_word))
        observable = qml.pauli.string_to_pauli_word(word)
        dressed = self.dressed_majorana_product(word, kept_wires, device.num_wires)
        expected = (psi.conj() @ dressed @ psi).real
        np.testing.assert_allclose(
            self.run(state, [], observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize("device_name", ["prepared_device", "prepared_product_device"])
    @pytest.mark.parametrize("traced_wires", [[1], [0], [1, 3], [0, 2], [2, 3]])
    def test_fermionic_partial_trace_is_the_qubit_one_after_moving_the_wires_behind(
        self, request, device_name, traced_wires
    ):
        device, psi, _ = request.getfixturevalue(device_name)
        n_wires = device.num_wires
        state = device.partial_trace(traced_wires, fermionic=True)
        swaps = self.move_behind(traced_wires, n_wires)
        unitary = np.asarray(qml.matrix(qml.tape.QuantumScript(swaps), wire_order=range(n_wires)))
        moved = unitary @ psi
        expected = np.asarray(
            qml.math.partial_trace(np.outer(moved, moved.conj()), list(range(n_wires - len(traced_wires), n_wires)))
        )
        np.testing.assert_allclose(
            state.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_non_contiguous_follow_up_circuit_matches_embedded_free_fermion_evolution(self, prepared_device):
        # Tracing out wires and running a matchgate circuit on the re-indexed remaining wires is the same as
        # applying that circuit to the Majorana modes of the kept wires in the full system.
        device, psi, rng = prepared_device
        traced_wires = [1, 3]
        state = device.partial_trace(traced_wires, fermionic=True)
        kept_wires = state.source_wires.tolist()
        follow_up = self.brick_wall(range(3), 1, rng)
        reduced_device = NonInteractingFermionicDevice(wires=3)
        reduced_device.apply(follow_up)
        reduced_sptm = np.asarray(reduced_device.global_sptm.sptm_matrix())
        embedded_sptm = np.eye(2 * device.num_wires)
        majorana_map = MixedGaussianState.majorana_indices(kept_wires)
        embedded_sptm[np.ix_(majorana_map, majorana_map)] = reduced_sptm
        unitary = np.asarray(SingleParticleTransitionMatrixOperation.to_unitary_matrix(torch.tensor(embedded_sptm)))
        evolved_psi = unitary @ psi
        for word in ["XXI", "IZZ", "YIX"]:
            observable = qml.pauli.string_to_pauli_word(word)
            dressed = self.dressed_majorana_product(word, kept_wires, device.num_wires)
            expected = (evolved_psi.conj() @ dressed @ evolved_psi).real
            np.testing.assert_allclose(
                self.run(state, follow_up, observable),
                expected,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )

    def test_partial_trace_after_qnode_execution(self):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        operations = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)

        @qml.qnode(device)
        def circuit():
            qml.BasisState(np.array([1, 1, 0, 0]), wires=range(n_wires))
            for operation in operations:
                qml.apply(operation)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = device.partial_trace([3])
        psi = self.dense_state(np.array([1, 1, 0, 0]), operations, n_wires)
        expected = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [3]))
        np.testing.assert_allclose(
            state.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_reduced_state_feeds_the_next_circuit_in_a_qnode(self, prepared_product_device):
        device, psi, rng = prepared_product_device
        state = device.partial_trace([3, 4])
        follow_up = self.brick_wall(range(3), 1, rng)
        observable = qml.PauliX(0) + 0.5 * qml.PauliZ(0) @ qml.PauliY(1) - 0.3 * qml.PauliX(1) @ qml.PauliX(2)

        @qml.qnode(qml.device("nif.qubit", wires=3))
        def circuit():
            state.state_prep()
            for operation in follow_up:
                qml.apply(operation)
            return qml.expval(observable), qml.probs(wires=[0, 1, 2])

        unitary = np.asarray(qml.matrix(qml.tape.QuantumScript(follow_up), wire_order=range(3)))
        reduced = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [3, 4]))
        evolved = unitary @ reduced @ unitary.conj().T
        value, probabilities = circuit()
        np.testing.assert_allclose(
            value,
            np.trace(evolved @ qml.matrix(observable, wire_order=range(3))).real,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            probabilities, np.diag(evolved).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_partial_trace_of_fresh_device_is_zero_state(self):
        device = NonInteractingFermionicDevice(wires=3)
        state = device.partial_trace([2])
        assert state.is_pure
        np.testing.assert_allclose(state.density_matrix(), np.diag([1.0, 0.0, 0.0, 0.0]), atol=ATOL_MATRIX_COMPARISON)

    def test_partial_trace_batched_device(self):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 4
        batched_angles = torch.tensor(rng.normal(size=(2, 2)), dtype=torch.float64)
        shared = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([CompRyRy(batched_angles, wires=[1, 2])] + shared)
        state = device.partial_trace([0])
        assert state.batch_shape == (2,)
        assert isinstance(state.covariance_matrix, torch.Tensor)
        for batch_index in range(2):
            operations = [CompRyRy(batched_angles[batch_index], wires=[1, 2])] + shared
            psi = self.dense_state(np.zeros(n_wires, dtype=int), operations, n_wires)
            expected = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [0]))
            np.testing.assert_allclose(
                state.density_matrix()[batch_index].numpy(),
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    def test_partial_trace_rejects_unknown_wires(self, prepared_device):
        device, _, _ = prepared_device
        with pytest.raises(ValueError, match="not wires of the device"):
            device.partial_trace([7])
        with pytest.raises(ValueError, match="not wires of the device"):
            device.reduced_state([0, 9])

    def test_reduced_state_rejects_empty_kept_wires(self, prepared_device):
        device, _, _ = prepared_device
        with pytest.raises(ValueError, match="At least one wire"):
            device.partial_trace(list(range(device.num_wires)))

    @pytest.mark.parametrize("traced_wires", [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]])
    def test_partial_trace_of_product_state_matches_qubit_partial_trace(self, prepared_product_device, traced_wires):
        device, psi, _ = prepared_product_device
        state = device.partial_trace(traced_wires)
        assert state.is_displaced
        expected = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), traced_wires))
        np.testing.assert_allclose(
            state.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_partial_trace_of_product_state_carries_the_displacement(self, prepared_product_device):
        device, _, _ = prepared_product_device
        state = device.reduced_state([0, 1])
        extended = np.asarray(device.extended_covariance_matrix)
        indices = np.array([0, 1, 2, 3])
        np.testing.assert_allclose(
            state.covariance_matrix,
            extended[np.ix_(indices, indices)],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            state.displacement, extended[indices, -1], atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("traced_wires", [[0], [0, 1], [0, 4]])
    def test_partial_trace_of_product_state_in_front_must_be_requested_as_fermionic(
        self, prepared_product_device, traced_wires
    ):
        device, psi, _ = prepared_product_device
        with pytest.raises(ValueError, match="fermionic=True"):
            device.partial_trace(traced_wires)
        state = device.partial_trace(traced_wires, fermionic=True)
        kept_wires = state.source_wires.tolist()
        reduced = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), traced_wires))
        # Probabilities and words with an even number of X and Y factors are those of the qubit partial trace.
        np.testing.assert_allclose(
            self.run(state, [], output_type="probs"),
            np.diag(reduced).real,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        for pauli_word in ["XY", "ZI", "YX", "XI", "ZY", "IX"]:
            word = pauli_word + "I" * (len(kept_wires) - len(pauli_word))
            observable = qml.pauli.string_to_pauli_word(word)
            dressed = self.dressed_majorana_product(word, kept_wires, device.num_wires)
            expected = (psi.conj() @ dressed @ psi).real
            np.testing.assert_allclose(
                self.run(state, [], observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            if (word.count("X") + word.count("Y")) % 2 == 0:
                qubit_expected = np.trace(reduced @ qml.matrix(observable, wire_order=range(len(kept_wires)))).real
                np.testing.assert_allclose(
                    expected, qubit_expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
                )

    @pytest.mark.parametrize("pauli_word", ["XI", "IX", "ZY", "XX", "YZ"])
    def test_non_contiguous_partial_trace_of_product_state_is_the_fermionic_one(
        self, prepared_product_device, pauli_word
    ):
        device, psi, _ = prepared_product_device
        state = device.reduced_state([0, 2], fermionic=True)
        observable = qml.pauli.string_to_pauli_word(pauli_word)
        dressed = self.dressed_majorana_product(pauli_word, [0, 2], device.num_wires)
        expected = (psi.conj() @ dressed @ psi).real
        np.testing.assert_allclose(
            self.run(state, [], observable), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_partial_trace_of_plus_state(self):
        n_wires = 3
        operations = [CompRxRx(np.array([0.1, 0.2]), wires=[0, 1]), CompRyRy(np.array([0.7, -0.4]), wires=[1, 2])]
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([PlusState(wires=range(n_wires))] + operations)
        state = device.partial_trace([2])
        psi = self.dense_product_state(np.full((n_wires, 2), np.sqrt(0.5)), operations, n_wires)
        expected = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [2]))
        assert state.is_displaced
        np.testing.assert_allclose(
            state.density_matrix(), expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        eigenvalues = np.linalg.eigvalsh(expected)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        np.testing.assert_allclose(
            state.entropy(),
            -np.sum(eigenvalues * np.log(eigenvalues)),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_partial_trace_of_batched_superposed_product_state(self):
        rng = np.random.default_rng(TEST_SEED)
        n_wires = 3
        amplitudes = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.2), 1j * np.sqrt(0.8)]],
            ]
        )
        operations = self.brick_wall(range(n_wires), 1, rng)
        device = NonInteractingFermionicDevice(wires=n_wires)
        device.apply([ProductState(amplitudes, wires=range(n_wires))] + operations)
        state = device.partial_trace([2])
        assert state.batch_shape == (2,)
        assert state.is_displaced
        observable = qml.PauliX(0) @ qml.PauliZ(1) + 0.4 * qml.PauliY(1)
        values = np.asarray(self.run(state, [], observable))
        for batch_index in range(2):
            psi = self.dense_product_state(amplitudes[batch_index], operations, n_wires)
            expected = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), [2]))
            np.testing.assert_allclose(
                np.asarray(state.density_matrix())[batch_index],
                expected,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )
            np.testing.assert_allclose(
                values[batch_index],
                np.trace(expected @ qml.matrix(observable, wire_order=range(2))).real,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )

    def test_partial_trace_of_product_state_keeps_device_dtype(self):
        device = NonInteractingFermionicDevice(wires=3, r_dtype=torch.float32, c_dtype=torch.complex64)
        device.apply([PlusState(wires=range(3)), CompRxRx(np.array([0.1, 0.2]), wires=[0, 1])])
        state = device.partial_trace([2])
        assert qml.math.get_dtype_name(state.covariance_matrix) == "float32"
        assert qml.math.get_dtype_name(state.displacement) == "float32"

    def test_partial_trace_accepts_batched_basis_product_state(self):
        device = NonInteractingFermionicDevice(wires=2)
        device.apply([ProductState.from_basis_state(np.array([[0, 1], [1, 0]]), wires=[0, 1])])
        state = device.partial_trace([0])
        assert state.batch_shape == (2,)
        density = np.asarray(state.density_matrix())
        np.testing.assert_allclose(density[0], np.diag([0.0, 1.0]), atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(density[1], np.diag([1.0, 0.0]), atol=ATOL_MATRIX_COMPARISON)

    def test_partial_trace_rejects_a_state_preparation_without_covariance(self):
        device = NonInteractingFermionicDevice(wires=2)
        device.apply_state_prep(qml.StatePrep(np.array([1.0, 0.0, 0.0, 0.0]), wires=[0, 1]), index=0)
        with pytest.raises(ValueError, match="covariance"):
            device.partial_trace([0])
