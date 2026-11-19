from abc import abstractmethod

from pennylane.operation import StatePrepBase
from pennylane.wires import WiresLike

from ...typing import TensorLike


class CovarianceStatePrepBase(StatePrepBase):
    r"""
    A state preparation described by a Majorana covariance matrix and a displacement vector.

    With the Jordan-Wigner Majorana operators :math:`c_{2k} = Z_{<k} X_k` and :math:`c_{2k+1} = Z_{<k} Y_k`
    of the wires of the operation, the state it prepares is characterized by

    .. math::
        \Lambda_{\mu\nu} = i\, \mathrm{Tr}[\rho\, c_\mu c_\nu] \;\; (\mu \neq \nu),
        \qquad
        d_\mu = \mathrm{Tr}[\rho\, c_\mu],

    and every expectation value follows from them through Wick's theorem in the extended Majorana algebra
    (see :func:`~matchcake.devices.expval_strategies.m_pfaffian._extended_covariance.extended_covariance_matrix`).
    This is all the :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` needs from its initial
    state: a matchgate circuit with single-particle transition matrix :math:`R` maps them to
    :math:`R^\top \Lambda R` and :math:`R^\top d`. A subclass provides :attr:`covariance_matrix`,
    :attr:`displacement` and :meth:`to_wires`, and the device then computes expectation values, probabilities
    and samples for it. :class:`~matchcake.operations.state_preparation.ProductState` and
    :class:`~matchcake.operations.state_preparation.GaussianStatePrep` are the two implementations.
    """

    @abstractmethod
    def to_wires(self, wires: WiresLike) -> "CovarianceStatePrepBase":
        r"""
        Return the same state on ``wires``, the wires it does not act on being in :math:`|0\rangle`.

        A device uses this to extend a state preparation on part of its wires to its full register.

        :param wires: The wires of the new operation. They must contain every wire of this operation.
        :type wires: WiresLike
        :return: The state preparation on ``wires``.
        :rtype: CovarianceStatePrepBase
        """

    @property
    @abstractmethod
    def covariance_matrix(self) -> TensorLike:
        """The Majorana covariance matrix of the prepared state, of shape ``(..., 2n, 2n)``.

        :return: The real antisymmetric covariance matrix.
        :rtype: TensorLike
        """

    @property
    @abstractmethod
    def displacement(self) -> TensorLike:
        """The displacement vector of the prepared state, of shape ``(..., 2n)``.

        It vanishes for a state with a definite fermion parity.

        :return: The real displacement vector.
        :rtype: TensorLike
        """
