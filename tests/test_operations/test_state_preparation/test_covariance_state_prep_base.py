import numpy as np
import pytest

from matchcake.operations.state_preparation import (
    CovarianceStatePrepBase,
    GaussianStatePrep,
    MinusState,
    PlusState,
    ProductState,
)

from ...configs import ATOL_MATRIX_COMPARISON


class TestCovarianceStatePrepBase:
    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            CovarianceStatePrepBase(np.zeros(2), wires=[0])  # type: ignore[abstract]

    @pytest.mark.parametrize(
        "operation",
        [
            ProductState(np.array([[1.0, 0.0], [0.0, 1.0]]), wires=[0, 1]),
            PlusState(wires=[0, 1]),
            MinusState(wires=[0, 1]),
            GaussianStatePrep(np.zeros((4, 4)), wires=[0, 1]),
        ],
    )
    def test_implementations_provide_the_covariance_and_the_displacement(self, operation):
        assert isinstance(operation, CovarianceStatePrepBase)
        assert tuple(np.shape(np.asarray(operation.covariance_matrix))) == (4, 4)
        assert tuple(np.shape(np.asarray(operation.displacement))) == (4,)
        extended = operation.to_wires([0, 1, 2])
        assert isinstance(extended, CovarianceStatePrepBase)
        assert extended.wires.tolist() == [0, 1, 2]

    def test_plus_state_displacement(self):
        # <c_0> = <X_0> = 1, and the string of the first wire cancels the displacement of the second.
        np.testing.assert_allclose(
            PlusState(wires=[0, 1]).displacement, [1.0, 0.0, 0.0, 0.0], atol=ATOL_MATRIX_COMPARISON
        )
