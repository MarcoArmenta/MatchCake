import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.operations import CompRxRx, CompRyRy
from matchcake.operations.state_preparation import CovarianceStatePrepBase, GaussianStatePrep, ProductState
from matchcake.utils.covariance import block_diagonal_covariance
from matchcake.utils.majorana import get_majorana

from ...configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED


class TestGaussianStatePrep:
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

    @staticmethod
    def dense_state(amplitudes: np.ndarray, operations) -> np.ndarray:
        """State vector of a circuit applied to a tensor product of single-qubit states.

        :param amplitudes: Single-qubit amplitudes of shape ``(n_wires, 2)``.
        :type amplitudes: np.ndarray
        :param operations: Operations of the circuit.
        :type operations: List[Operation]
        :return: The state vector.
        :rtype: np.ndarray
        """
        n_wires = len(amplitudes)
        psi = np.ones(1, dtype=complex)
        for qubit_amplitudes in amplitudes:
            psi = np.kron(psi, qubit_amplitudes)
        unitary = np.asarray(qml.matrix(qml.tape.QuantumScript(operations), wire_order=range(n_wires)))
        return unitary @ psi

    def test_is_a_covariance_state_prep(self):
        covariance = block_diagonal_covariance(np.array([0.5, -0.2]))
        operation = GaussianStatePrep(covariance, wires=[0, 1])
        assert isinstance(operation, CovarianceStatePrepBase)
        assert isinstance(operation, qml.operation.StatePrepBase)
        assert operation.wires.tolist() == [0, 1]
        assert operation.num_params == 2
        assert operation.batch_size is None
        np.testing.assert_allclose(operation.covariance_matrix, covariance, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(operation.displacement, np.zeros(4), atol=ATOL_MATRIX_COMPARISON)

    def test_keeps_the_displacement(self):
        covariance = np.zeros((2, 2))
        operation = GaussianStatePrep(covariance, np.array([0.6, -0.3]), wires=[4])
        np.testing.assert_allclose(operation.displacement, [0.6, -0.3], atol=ATOL_MATRIX_COMPARISON)

    def test_batched_covariance_gives_a_batch_size(self):
        covariance = block_diagonal_covariance(np.array([[0.5, -0.2], [1.0, 0.0], [0.1, 0.3]]))
        operation = GaussianStatePrep(covariance, wires=[0, 1])
        assert operation.batch_size == 3
        assert tuple(qml.math.shape(operation.covariance_matrix)) == (3, 4, 4)

    def test_keeps_the_backend_and_the_gradient(self):
        covariance = torch.tensor(block_diagonal_covariance(np.array([0.5, -0.2])), requires_grad=True)
        operation = GaussianStatePrep(covariance, wires=[0, 1])
        assert isinstance(operation.covariance_matrix, torch.Tensor)
        assert operation.covariance_matrix.requires_grad

    def test_rejects_missing_wires(self):
        with pytest.raises(ValueError, match="wires"):
            GaussianStatePrep(np.zeros((4, 4)))

    @pytest.mark.parametrize("shape", [(3, 3), (4, 6), (6, 6), (2, 3, 4, 4)])
    def test_rejects_a_covariance_of_the_wrong_shape(self, shape):
        with pytest.raises(ValueError, match="shape"):
            GaussianStatePrep(np.zeros(shape), wires=[0, 1])

    def test_rejects_a_displacement_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match="displacement"):
            GaussianStatePrep(np.zeros((4, 4)), np.zeros(3), wires=[0, 1])

    def test_rejects_a_matrix_that_is_not_antisymmetric(self):
        with pytest.raises(ValueError, match="antisymmetric"):
            GaussianStatePrep(np.ones((4, 4)), wires=[0, 1])
        operation = GaussianStatePrep(np.ones((4, 4)), wires=[0, 1], validate=False)
        assert operation.wires.tolist() == [0, 1]

    def test_to_wires_returns_itself_on_its_own_wires(self):
        operation = GaussianStatePrep(block_diagonal_covariance(np.array([0.5, -0.2])), wires=[0, 1])
        assert operation.to_wires([0, 1]) is operation

    @pytest.mark.parametrize("wires, register", [([0, 2], [0, 1, 2]), ([1, 2], [0, 1, 2, 3]), ([0, 1], [0, 1, 2])])
    def test_to_wires_puts_the_other_wires_in_the_zero_state(self, wires, register):
        rng = np.random.default_rng(TEST_SEED)
        amplitudes = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        amplitudes /= np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        operations = [CompRxRx(rng.normal(size=2), wires=[0, 1]), CompRyRy(rng.normal(size=2), wires=[0, 1])]
        psi = self.dense_state(amplitudes, operations).reshape(2, 2)
        covariance, displacement = self.dense_covariance_and_displacement(
            np.outer(psi.reshape(-1), psi.reshape(-1).conj()), 2
        )
        extended = GaussianStatePrep(covariance, displacement, wires=wires).to_wires(register)
        assert extended.wires.tolist() == register
        # Dense state of the register: the two wires hold psi, the others are in the zero state.
        full = np.zeros([2] * len(register), dtype=complex)
        index = [0] * len(register)
        for first in range(2):
            for second in range(2):
                index[register.index(wires[0])] = first
                index[register.index(wires[1])] = second
                full[tuple(index)] = psi[first, second]
        full = full.reshape(-1)
        expected_covariance, expected_displacement = self.dense_covariance_and_displacement(
            np.outer(full, full.conj()), len(register)
        )
        np.testing.assert_allclose(
            extended.covariance_matrix, expected_covariance, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        np.testing.assert_allclose(
            extended.displacement, expected_displacement, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_to_wires_keeps_a_batch(self):
        covariance = block_diagonal_covariance(np.array([[0.5], [-1.0]]))
        extended = GaussianStatePrep(covariance, wires=[1]).to_wires([0, 1])
        assert extended.batch_size == 2
        np.testing.assert_allclose(
            extended.covariance_matrix,
            block_diagonal_covariance(np.array([[1.0, 0.5], [1.0, -1.0]])),
            atol=ATOL_MATRIX_COMPARISON,
        )

    def test_to_wires_rejects_missing_wires_and_another_order(self):
        operation = GaussianStatePrep(block_diagonal_covariance(np.array([0.5, -0.2])), wires=[0, 1])
        with pytest.raises(ValueError, match="do not contain"):
            operation.to_wires([1, 2])
        with pytest.raises(ValueError, match="order"):
            operation.to_wires([1, 0, 2])

    def test_state_vector_is_not_defined(self):
        operation = GaussianStatePrep(block_diagonal_covariance(np.array([0.5, -0.2])), wires=[0, 1])
        with pytest.raises(NotImplementedError, match="state vector"):
            operation.state_vector()

    def test_parameters_can_be_rebound(self):
        operation = GaussianStatePrep(block_diagonal_covariance(np.array([0.5, -0.2])), np.zeros(4), wires=[0, 1])
        new_covariance = block_diagonal_covariance(np.array([1.0, 1.0]))
        rebound = qml.ops.functions.bind_new_parameters(operation, [new_covariance, np.zeros(4)])
        assert isinstance(rebound, GaussianStatePrep)
        assert rebound.wires == operation.wires
        np.testing.assert_allclose(rebound.covariance_matrix, new_covariance, atol=ATOL_MATRIX_COMPARISON)

    def test_product_state_is_a_covariance_state_prep_with_its_displacement(self):
        rng = np.random.default_rng(TEST_SEED + 1)
        amplitudes = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        amplitudes /= np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        operation = ProductState(amplitudes, wires=range(3))
        assert isinstance(operation, CovarianceStatePrepBase)
        psi = self.dense_state(amplitudes, [])
        _, expected = self.dense_covariance_and_displacement(np.outer(psi, psi.conj()), 3)
        np.testing.assert_allclose(
            operation.displacement, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        batched = ProductState(np.stack([amplitudes, np.eye(2)[[0, 1, 1]]]), wires=range(3))
        assert tuple(qml.math.shape(batched.displacement)) == (2, 6)
        np.testing.assert_allclose(batched.displacement[0], expected, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(batched.displacement[1], np.zeros(6), atol=ATOL_MATRIX_COMPARISON)
