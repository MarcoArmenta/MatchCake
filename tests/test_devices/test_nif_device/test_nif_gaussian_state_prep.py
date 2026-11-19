import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake import NonInteractingFermionicDevice
from matchcake.operations import CompRxRx, CompRyRy, CompRzRz, fSWAP
from matchcake.operations.state_preparation import GaussianStatePrep
from matchcake.utils.majorana import get_majorana

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    GRADCHECK_ATOL,
    GRADCHECK_EPS,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)

N_SHOTS = 20_000


class TestNonInteractingFermionicDeviceGaussianStatePrep:
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
    def dense_covariance_and_displacement(density_matrix: np.ndarray, n_wires: int):
        """Brute-force ``i Tr[rho c_mu c_nu]`` and ``Tr[rho c_mu]`` of a density matrix.

        :param density_matrix: Density matrix of shape ``(2^n, 2^n)``.
        :type density_matrix: np.ndarray
        :param n_wires: Number of wires.
        :type n_wires: int
        :return: The covariance matrix of shape ``(2n, 2n)`` and the displacement of shape ``(2n,)``.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        majoranas = [get_majorana(mu, n_wires) for mu in range(2 * n_wires)]
        covariance = np.zeros((2 * n_wires, 2 * n_wires))
        for mu in range(2 * n_wires):
            for nu in range(2 * n_wires):
                if mu != nu:
                    covariance[mu, nu] = (1j * np.trace(density_matrix @ majoranas[mu] @ majoranas[nu])).real
        displacement = np.array([np.trace(density_matrix @ majorana).real for majorana in majoranas])
        return covariance, displacement

    @classmethod
    def mixed_state(cls, superposed: bool, seed: int, n_kept: int = 3):
        """A mixed Gaussian state of ``n_kept`` wires with its dense density matrix.

        It is the reduced state, on the first wires, of a random matchgate circuit on ``n_kept + 2`` wires
        that starts from a basis state, or from a product state with superposed qubits when ``superposed``.
        The covariance matrix and the displacement are computed by brute force from the dense state.

        :param superposed: Whether the circuit starts from a product state with superposed qubits.
        :type superposed: bool
        :param seed: Seed of the random generator.
        :type seed: int
        :param n_kept: Number of wires of the state.
        :type n_kept: int
        :return: The covariance matrix, the displacement and the density matrix.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        rng = np.random.default_rng(seed)
        n_wires = n_kept + 2
        amplitudes = np.eye(2)[rng.integers(0, 2, size=n_wires)].astype(complex)
        if superposed:
            amplitudes = rng.normal(size=(n_wires, 2)) + 1j * rng.normal(size=(n_wires, 2))
            amplitudes /= np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        psi = np.ones(1, dtype=complex)
        for qubit_amplitudes in amplitudes:
            psi = np.kron(psi, qubit_amplitudes)
        psi = cls.dense_unitary(cls.brick_wall(range(n_wires), 2, rng), n_wires) @ psi
        density = np.asarray(qml.math.partial_trace(np.outer(psi, psi.conj()), list(range(n_kept, n_wires))))
        covariance, displacement = cls.dense_covariance_and_displacement(density, n_kept)
        return covariance, displacement, density

    @pytest.fixture(params=[False, True], ids=["even", "displaced"])
    def prepared_case(self, request):
        """A mixed state of three wires, a follow-up circuit and the dense evolved density matrix."""
        covariance, displacement, density = self.mixed_state(request.param, TEST_SEED + 30)
        rng = np.random.default_rng(TEST_SEED + 31)
        follow_up = self.brick_wall(range(3), 1, rng)
        unitary = self.dense_unitary(follow_up, 3)
        return covariance, displacement, follow_up, unitary @ density @ unitary.conj().T

    def test_qnode_expval_matches_dense(self, prepared_case):
        covariance, displacement, follow_up, evolved_density = prepared_case
        observable = qml.PauliX(0) + 0.5 * qml.PauliZ(0) @ qml.PauliY(1) - 0.3 * qml.PauliX(1) @ qml.PauliX(2)

        @qml.qnode(qml.device("nif.qubit", wires=3))
        def circuit():
            GaussianStatePrep(covariance, displacement, wires=range(3))
            for operation in follow_up:
                qml.apply(operation)
            return qml.expval(observable)

        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
        np.testing.assert_allclose(circuit(), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_qnode_probs_match_dense(self, prepared_case):
        covariance, displacement, follow_up, evolved_density = prepared_case

        @qml.qnode(qml.device("nif.qubit", wires=3))
        def circuit(wires):
            GaussianStatePrep(covariance, displacement, wires=range(3))
            for operation in follow_up:
                qml.apply(operation)
            return qml.probs(wires=wires)

        np.testing.assert_allclose(
            circuit([0, 1, 2]), np.diag(evolved_density).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        marginal = np.asarray(qml.math.partial_trace(evolved_density, [1]))
        np.testing.assert_allclose(
            circuit([0, 2]), np.diag(marginal).real, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("sampling_strategy", ["ProductBasisSampling", "2QubitBy2QubitSampling"])
    def test_samples_follow_the_probabilities(self, prepared_case, sampling_strategy):
        covariance, displacement, follow_up, evolved_density = prepared_case
        np.random.seed(TEST_SEED)
        torch.manual_seed(TEST_SEED)
        device = NonInteractingFermionicDevice(wires=3, sampling_strategy=sampling_strategy)
        samples = np.asarray(
            device.execute_generator(
                [GaussianStatePrep(covariance, displacement, wires=range(3))] + follow_up,
                output_type="samples",
                shots=N_SHOTS,
                reset=True,
            )
        ).astype(int)
        assert samples.shape == (N_SHOTS, 3)
        frequencies = np.bincount(samples @ (2 ** np.arange(3)[::-1]), minlength=8) / N_SHOTS
        np.testing.assert_allclose(frequencies, np.diag(evolved_density).real, atol=ATOL_APPROX_COMPARISON)

    def test_shots_expval_of_an_observable_outside_the_computational_basis(self, prepared_case):
        covariance, displacement, follow_up, evolved_density = prepared_case
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        observable = qml.PauliX(0) @ qml.PauliY(1)

        @qml.set_shots(N_SHOTS)
        @qml.qnode(qml.device("nif.qubit", wires=3))
        def circuit():
            GaussianStatePrep(covariance, displacement, wires=range(3))
            for operation in follow_up:
                qml.apply(operation)
            return qml.expval(observable)

        expected = np.trace(evolved_density @ qml.matrix(observable, wire_order=range(3))).real
        np.testing.assert_allclose(circuit(), expected, atol=ATOL_APPROX_COMPARISON)

    def test_batch_of_covariances_matches_dense(self):
        states = [self.mixed_state(True, TEST_SEED + 40 + index) for index in range(3)]
        covariance = np.stack([state[0] for state in states])
        displacement = np.stack([state[1] for state in states])
        follow_up = self.brick_wall(range(3), 1, np.random.default_rng(TEST_SEED + 41))
        unitary = self.dense_unitary(follow_up, 3)
        observable = qml.PauliY(0) + 0.7 * qml.PauliZ(1) @ qml.PauliZ(2)
        device = NonInteractingFermionicDevice(wires=3)
        values = device.execute_generator(
            [GaussianStatePrep(covariance, displacement, wires=range(3))] + follow_up,
            observable=observable,
            output_type="expval",
            reset=True,
        )
        probabilities = device.execute_output(output_type="probs")
        assert tuple(qml.math.shape(values)) == (3,)
        for index, (_, _, density) in enumerate(states):
            evolved = unitary @ density @ unitary.conj().T
            expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(3))).real
            np.testing.assert_allclose(
                values[index], expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            np.testing.assert_allclose(
                np.asarray(probabilities)[index], np.diag(evolved).real, atol=ATOL_MATRIX_COMPARISON
            )

    def test_state_prep_on_part_of_the_wires_matches_dense(self):
        covariance, displacement, density = self.mixed_state(True, TEST_SEED + 50)
        n_wires = 5
        follow_up = self.brick_wall(range(n_wires), 1, np.random.default_rng(TEST_SEED + 51))
        zero = np.diag([1.0, 0.0])
        full_density = np.kron(np.kron(zero, density), zero)
        unitary = self.dense_unitary(follow_up, n_wires)
        evolved = unitary @ full_density @ unitary.conj().T
        observable = qml.PauliX(0) @ qml.PauliX(1) + 0.4 * qml.PauliY(2) @ qml.PauliZ(4) - qml.PauliZ(3)
        device = NonInteractingFermionicDevice(wires=n_wires)
        value = device.execute_generator(
            [GaussianStatePrep(covariance, displacement, wires=[1, 2, 3])] + follow_up,
            observable=observable,
            output_type="expval",
            reset=True,
        )
        expected = np.trace(evolved @ qml.matrix(observable, wire_order=range(n_wires))).real
        np.testing.assert_allclose(value, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_gradient_with_respect_to_the_covariance(self):
        covariance, displacement, _ = self.mixed_state(True, TEST_SEED + 60)
        follow_up = self.brick_wall(range(3), 1, np.random.default_rng(TEST_SEED + 61))
        observable = qml.PauliX(0) + qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(2)
        device = NonInteractingFermionicDevice(wires=3)

        def expectation(raw_covariance, raw_displacement):
            antisymmetric = 0.5 * (raw_covariance - raw_covariance.mT)
            return device.execute_generator(
                [GaussianStatePrep(antisymmetric, raw_displacement, wires=range(3))] + follow_up,
                observable=observable,
                output_type="expval",
                reset=True,
            )

        inputs = (
            torch.tensor(covariance, dtype=torch.float64, requires_grad=True),
            torch.tensor(displacement, dtype=torch.float64, requires_grad=True),
        )
        assert torch.autograd.gradcheck(
            expectation, inputs, eps=GRADCHECK_EPS, atol=GRADCHECK_ATOL, raise_exception=True
        )

    def test_qnode_gradient_with_respect_to_the_covariance_and_the_circuit(self):
        covariance, displacement, density = self.mixed_state(False, TEST_SEED + 70)
        observable = qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(1) @ qml.PauliX(2)

        @qml.qnode(qml.device("nif.qubit", wires=3), interface="torch", diff_method="backprop")
        def circuit(covariance_matrix, angles):
            GaussianStatePrep(covariance_matrix, wires=range(3))
            CompRxRx(angles, wires=[0, 1])
            CompRyRy(angles, wires=[1, 2])
            return qml.expval(observable)

        covariance_t = torch.tensor(covariance, dtype=torch.float64, requires_grad=True)
        angles = torch.tensor([0.3, -0.8], dtype=torch.float64, requires_grad=True)
        value = circuit(covariance_t, angles)
        gradient_covariance, gradient_angles = torch.autograd.grad(value, (covariance_t, angles))
        assert gradient_covariance.shape == (6, 6)

        reference_angles = angles.detach().clone().requires_grad_()
        operations = [CompRxRx(reference_angles, wires=[0, 1]), CompRyRy(reference_angles, wires=[1, 2])]
        unitary = qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(3))
        evolved = unitary @ torch.tensor(density) @ unitary.conj().T
        reference = torch.trace(evolved @ torch.tensor(qml.matrix(observable, wire_order=range(3)))).real
        (reference_gradient,) = torch.autograd.grad(reference, reference_angles)
        torch.testing.assert_close(
            value.detach(), reference.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            gradient_angles, reference_gradient, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_covariance_matrix_of_the_device_evolves_the_prepared_one(self, prepared_case):
        covariance, displacement, follow_up, evolved_density = prepared_case
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([GaussianStatePrep(covariance, displacement, wires=range(3))] + follow_up)
        expected_covariance, expected_displacement = self.dense_covariance_and_displacement(evolved_density, 3)
        np.testing.assert_allclose(
            device.covariance_matrix, expected_covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        extended = np.asarray(device.extended_covariance_matrix)
        np.testing.assert_allclose(
            extended[:-1, -1], expected_displacement, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_pure_gaussian_state_prep_matches_the_circuit_that_made_it(self):
        rng = np.random.default_rng(TEST_SEED + 80)
        first = self.brick_wall(range(3), 2, rng)
        second = self.brick_wall(range(3), 1, rng)
        observable = qml.PauliX(0) @ qml.PauliY(1) + qml.PauliZ(2)
        device = NonInteractingFermionicDevice(wires=3)
        device.apply([qml.BasisState(np.array([1, 0, 1]), wires=range(3))] + first)
        covariance = device.covariance_matrix
        direct = device.execute_generator(
            [qml.BasisState(np.array([1, 0, 1]), wires=range(3))] + first + second,
            observable=observable,
            output_type="expval",
            reset=True,
        )
        prepared = device.execute_generator(
            [GaussianStatePrep(covariance, wires=range(3))] + second,
            observable=observable,
            output_type="expval",
            reset=True,
        )
        np.testing.assert_allclose(prepared, direct, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
