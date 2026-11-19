from typing import Any, Optional

import numpy as np
import pennylane as qml
import torch
from pennylane.wires import Wires, WiresLike

from ...typing import TensorLike
from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import infer_real_dtype
from .covariance_state_prep_base import CovarianceStatePrepBase


class GaussianStatePrep(CovarianceStatePrepBase):
    r"""
    Prepare a fermionic Gaussian state, pure or mixed, from its Majorana covariance matrix.

    The state is given by the real antisymmetric matrix
    :math:`\Lambda_{\mu\nu} = i\, \mathrm{Tr}[\rho\, c_\mu c_\nu]` of its wires and, optionally, by a displacement
    vector :math:`d_\mu = \mathrm{Tr}[\rho\, c_\mu]`. This covers the states that a matchgate circuit can start
    from but that are not product states: the reduced state of a previous circuit on some of its wires
    (see :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.partial_trace`), a thermal state of a
    free-fermion Hamiltonian, or a mixture of occupation numbers. The
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` evolves the covariance matrix and the
    displacement linearly, so expectation values, probabilities, samples and gradients with respect to
    :math:`\Lambda` come from the usual pipeline of the device, inside or outside a QNode.

    A leading batch dimension, ``(B, 2n, 2n)``, prepares a batch of states.

    :param covariance_matrix: Real antisymmetric matrix of shape ``(2n, 2n)`` or ``(B, 2n, 2n)``.
    :type covariance_matrix: TensorLike
    :param displacement: Real displacement vector of shape ``(2n,)`` or ``(B, 2n)``. Defaults to ``None``
        (no displacement, the state has a definite fermion parity).
    :type displacement: Optional[TensorLike]
    :param wires: The ``n`` wires of the state.
    :type wires: WiresLike
    :param validate: Whether to check that the covariance matrix is antisymmetric. Defaults to ``True``.
    :type validate: bool
    """

    num_wires = None
    grad_method = None
    ANTISYMMETRY_ATOL = 1e-6

    def __init__(
        self,
        covariance_matrix: TensorLike,
        displacement: Optional[TensorLike] = None,
        wires: Optional[WiresLike] = None,
        validate: bool = True,
    ):
        if wires is None:
            raise ValueError(f"The wires of a {type(self).__name__} must be given.")
        wires = Wires(wires)
        num_majoranas = 2 * len(wires)
        shape = tuple(qml.math.shape(covariance_matrix))
        if len(shape) not in (2, 3) or shape[-2:] != (num_majoranas, num_majoranas):
            raise ValueError(
                f"The covariance matrix of {len(wires)} wires must have shape ({num_majoranas}, {num_majoranas}) "
                f"or (B, {num_majoranas}, {num_majoranas}), got shape {shape}."
            )
        if displacement is None:
            displacement = convert_and_cast_like(np.zeros(num_majoranas), covariance_matrix)
        displacement_shape = tuple(qml.math.shape(displacement))
        if len(displacement_shape) not in (1, 2) or displacement_shape[-1] != num_majoranas:
            raise ValueError(
                f"The displacement of {len(wires)} wires must have shape ({num_majoranas},) or "
                f"(B, {num_majoranas}), got shape {displacement_shape}."
            )
        if validate:
            transposed = qml.math.einsum("...ij->...ji", covariance_matrix)
            if not qml.math.allclose(covariance_matrix, -transposed, atol=self.ANTISYMMETRY_ATOL):
                raise ValueError("The covariance matrix must be real antisymmetric.")
        super().__init__(covariance_matrix, displacement, wires=wires)

    def state_vector(self, wire_order: Optional[WiresLike] = None) -> TensorLike:
        """A Gaussian state given by its covariance matrix is generally mixed and has no state vector.

        :param wire_order: Unused.
        :type wire_order: Optional[WiresLike]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"A {type(self).__name__} holds a covariance matrix, which may describe a mixed state, and has no "
            f"state vector. Run it on a NonInteractingFermionicDevice."
        )

    def to_wires(self, wires: WiresLike) -> "GaussianStatePrep":
        r"""
        Return the same state on ``wires``, the wires it does not act on being in :math:`|0\rangle`.

        A wire in :math:`|0\rangle` has no displacement and an even parity, so it does not change the
        Jordan-Wigner strings that run through it: the covariance matrix and the displacement of this state
        are placed on the Majorana indices of its wires, and each new wire adds the block of
        :math:`|0\rangle`. The wires of this state must appear in ``wires`` in the same order, since a
        permutation of the wires of a Gaussian state is not a permutation of its Majorana operators.

        :param wires: The wires of the new operation. They must contain every wire of this operation, in the
            same order.
        :type wires: WiresLike
        :return: This operation if ``wires`` already matches its wires, otherwise a new one on ``wires``.
        :rtype: GaussianStatePrep
        :raises ValueError: If ``wires`` misses a wire of this state, or lists them in another order.
        """
        wires = Wires(wires)
        if wires == self.wires:
            return self
        missing_wires = [wire for wire in self.wires if wire not in wires]
        if missing_wires:
            raise ValueError(f"The wires {wires.tolist()} do not contain the wires {missing_wires} of {self}.")
        positions = [wires.index(wire) for wire in self.wires]
        if positions != sorted(positions):
            raise ValueError(
                f"The wires {self.wires.tolist()} of {type(self).__name__} must keep their order in "
                f"{wires.tolist()}: reordering the wires of a Gaussian state is not a relabeling of its "
                f"Majorana operators."
            )
        covariance_matrix: Any = self.covariance_matrix
        displacement: Any = self.displacement
        real_dtype = infer_real_dtype(covariance_matrix)
        covariance = torch.as_tensor(covariance_matrix, dtype=real_dtype)  # (..., 2n, 2n)
        size = 2 * len(wires)
        indices = np.stack([2 * np.asarray(positions), 2 * np.asarray(positions) + 1], axis=-1).reshape(-1)
        other_positions = np.asarray([position for position in range(len(wires)) if position not in positions])
        extended = covariance.new_zeros(tuple(covariance.shape[:-2]) + (size, size))
        extended[..., indices[:, None], indices[None, :]] = covariance
        # A wire in the zero state: Lambda_{2k, 2k+1} = -<Z_k> = -1.
        extended[..., 2 * other_positions, 2 * other_positions + 1] = -1.0
        extended[..., 2 * other_positions + 1, 2 * other_positions] = 1.0
        displacement_t = torch.as_tensor(displacement, dtype=real_dtype, device=covariance.device)  # (..., 2n)
        extended_displacement = displacement_t.new_zeros(tuple(displacement_t.shape[:-1]) + (size,))
        extended_displacement[..., indices] = displacement_t
        return GaussianStatePrep(
            convert_and_cast_like(extended, covariance_matrix),
            convert_and_cast_like(extended_displacement, covariance_matrix),
            wires=wires,
            validate=False,
        )

    @property
    def covariance_matrix(self) -> TensorLike:
        """The Majorana covariance matrix of the prepared state, of shape ``(..., 2n, 2n)``.

        :return: The covariance matrix.
        :rtype: TensorLike
        """
        return self.data[0]

    @property
    def displacement(self) -> TensorLike:
        """The displacement vector of the prepared state, of shape ``(..., 2n)``.

        :return: The displacement vector.
        :rtype: TensorLike
        """
        return self.data[1]

    @property
    def ndim_params(self) -> tuple:
        """The number of dimensions of the covariance matrix and of the displacement, without batch.

        :return: ``(2, 1)``.
        :rtype: tuple
        """
        return (2, 1)

    @property
    def num_params(self) -> int:
        """The number of parameters: the covariance matrix and the displacement.

        :return: ``2``.
        :rtype: int
        """
        return 2
