import itertools
import operator
from functools import cached_property, reduce
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import Operation, Operator, TermsUndefinedError
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.pauli import pauli_word_to_string
from pennylane.wires import Wires, WiresLike

from ..devices.expval_strategies.m_pfaffian._extended_covariance import extended_covariance_matrix, sptm_lift
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from ..operations.state_preparation.gaussian_state_prep import GaussianStatePrep
from ..operations.state_preparation.product_state import ProductState
from ..typing import TensorLike
from ..utils._pfaffian import sector_pfaffian_features
from ..utils.covariance import block_diagonal_covariance, block_diagonalize_covariance
from ..utils.majorana import MajoranaGetter
from ..utils.math import complex_dtype_name_like, convert_and_cast_like, convert_like_and_cast_to
from ..utils.torch_utils import detach, infer_real_dtype


class MixedGaussianState:
    r"""
    A fermionic Gaussian state, possibly mixed, held by its Majorana covariance matrix.

    The state lives on ``m`` wires and is fully characterized by the real antisymmetric matrix
    :math:`\Lambda_{\mu\nu} = i\,\mathrm{Tr}[\rho\, c_\mu c_\nu]` (:math:`\mu \neq \nu`) built from the
    Jordan-Wigner Majorana operators :math:`c_{2k} = Z_{<k} X_k` and :math:`c_{2k+1} = Z_{<k} Y_k` of the
    wires of the state. Such a state arises, for instance, as the reduced state of a matchgate circuit on a
    subset of its wires: its covariance matrix is the principal submatrix of the circuit's covariance matrix on
    the Majorana indices of the kept wires.

    This class is a container. It holds the state, computes what follows from its normal modes
    (polarizations, entropy, purity, pure-state decomposition), reduces it to fewer wires
    (:meth:`reduced_state`, :meth:`partial_trace`) and extends it with ancilla wires (:meth:`insert`,
    :meth:`append`). It does not run circuits: to evolve or measure the state, prepare it on a
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` with :meth:`state_prep`, which returns
    a :class:`~matchcake.operations.state_preparation.GaussianStatePrep`. Expectation values, probabilities,
    samples and gradients then come from the device, inside or outside a QNode, and
    :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.reduced_state` gives the evolved state
    back as a :class:`MixedGaussianState`.

    **Normal modes.** Block-diagonalizing the covariance matrix with
    :func:`~matchcake.utils.covariance.block_diagonalize_covariance`,

    .. math::
        \Lambda = R^\top \Big(\bigoplus_{j} (-\lambda_j) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\Big) R,

    exhibits the state as the product of single-mode mixtures
    :math:`\rho_j = \tfrac{1 + \lambda_j}{2} |0\rangle\langle 0| + \tfrac{1 - \lambda_j}{2} |1\rangle\langle 1|`
    evolved by the Gaussian unitary :math:`V_R` whose single-particle transition matrix is :math:`R`. The
    polarizations :math:`\lambda_j \in [-1, 1]` give the entropy and the purity. Expanding the product over
    the computational basis gives the **pure-state decomposition**

    .. math::
        \rho = \sum_{x} w_x\, V_R |x\rangle\langle x| V_R^\dagger,
        \qquad
        w_x = \prod_j \frac{1 + (-1)^{x_j} \lambda_j}{2},

    where each member :math:`V_R|x\rangle` is a state the device can prepare with a
    :class:`~matchcake.operations.state_preparation.ProductState` followed by a
    :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`. Modes with :math:`|\lambda_j| = 1`
    are pure and contribute a single term, so the sum runs over :math:`2^{r}` members where :math:`r` is the
    number of mixed modes (at most the number of traced-out wires). The members and their weights are exposed
    by :meth:`pure_states`, :attr:`basis_states`, :attr:`weights` and :meth:`ensemble_operations`. They are a
    way to analyze the state or to cross-check a result, not the way to run a circuit on it: the device
    evolves the covariance matrix itself at the cost of a single execution.

    The covariance matrix may carry a leading batch dimension, ``(B, 2m, 2m)``, for instance when the
    circuit that produced it had batched parameters. The batch is carried through every property and method.

    **Displaced states.** A circuit that starts from a product state with superposed qubits does not have a
    definite fermion parity, and its reduced states carry a displacement vector
    :math:`d_\mu = \mathrm{Tr}[\rho\, c_\mu]` next to the covariance matrix. Such a state is Gaussian in the
    extended algebra of the :math:`2m + 1` operators :math:`(c_0, \ldots, c_{2m-1}, P)`, with :math:`P` the
    parity of the wires of the state: every expectation value follows from the extended covariance matrix

    .. math::
        \tilde\Lambda = \begin{pmatrix} \Lambda & d \\ -d^\top & 0 \end{pmatrix}
        = \tilde R^\top \big(\Lambda_D \oplus 0\big) \tilde R,
        \qquad
        \tilde R \in SO(2m + 1),

    whose normal modes give the polarizations, the entropy and the weights :math:`w_x` exactly as above. The
    rotation :math:`\tilde R` mixes the Majorana operators with the parity, so the members of the pure-state
    decomposition are no longer basis states evolved by a matchgate circuit on the ``m`` wires. They are on
    ``m + 1`` wires: with one **parity wire** placed in front of the wires of the state (:attr:`parity_wire`,
    :attr:`ensemble_wires`), :math:`\tilde R \oplus 1` is an ordinary transition matrix, an operator with an
    even number of :math:`X` and :math:`Y` factors is measured as it is and an operator with an odd number of
    them is measured together with an :math:`X` on the parity wire (:meth:`ensemble_observable`).

    **Adding wires.** :meth:`insert` and :meth:`append` add ancilla wires in a product state at any position
    of the wire list, as ``list.insert`` and ``list.append`` do, and return the tensor product as a new state.
    They are the reverse of :meth:`partial_trace`.

    :param covariance_matrix: Real antisymmetric matrix of shape ``(2m, 2m)`` or ``(B, 2m, 2m)``.
    :type covariance_matrix: TensorLike
    :param wires: Labels of the ``m`` wires of the state. They must be consecutive integers because
        matchgates act on nearest-neighbour wires. Defaults to ``range(m)``.
    :type wires: Optional[WiresLike]
    :param displacement: Real displacement vector :math:`d_\mu = \mathrm{Tr}[\rho\, c_\mu]` of shape ``(2m,)``
        or ``(B, 2m)``, matching the batch of the covariance matrix. Defaults to ``None`` (no displacement).
    :type displacement: Optional[TensorLike]
    :param atol: A mode whose polarization satisfies :math:`|\lambda_j| \geq 1 - \text{atol}` is treated
        as pure and contributes a single member to the decomposition. Defaults to :attr:`DEFAULT_ATOL`.
    :type atol: float
    :param source_wires: Labels, in the original system, of the wires this state was reduced to. Purely
        informative; defaults to ``wires``.
    :type source_wires: Optional[WiresLike]
    """

    DEFAULT_ATOL = 1e-8
    ANTISYMMETRY_ATOL = 1e-6
    DISPLACEMENT_ATOL = 1e-12
    PHYSICALITY_ATOL = 1e-6

    @staticmethod
    def majorana_indices(wire_positions: Iterable[int]) -> np.ndarray:
        """Return the Majorana indices ``(2k, 2k+1)`` of every wire position, in order.

        :param wire_positions: Positions of the wires in the wire list of the state.
        :type wire_positions: Iterable[int]
        :return: Integer array of shape ``(2 * len(wire_positions),)``.
        :rtype: np.ndarray
        """
        positions = np.asarray(list(wire_positions), dtype=int)
        return np.stack([2 * positions, 2 * positions + 1], axis=-1).reshape(-1)

    @staticmethod
    def _entropy_term(probability: TensorLike) -> TensorLike:
        """Return ``p log p`` with the convention ``0 log 0 = 0``.

        :param probability: Probabilities in ``[0, 1]``.
        :type probability: TensorLike
        :return: The term, elementwise.
        :rtype: TensorLike
        """
        values: Any = probability
        positive = values > 0
        safe = qml.math.where(positive, values, qml.math.ones_like(values))
        return qml.math.where(positive, values * qml.math.log(safe), qml.math.zeros_like(values))

    @staticmethod
    def _ancilla_product_state(ancilla: Any) -> ProductState:
        """Return an ancilla given in any accepted form as a product state on ``range(k)``.

        :param ancilla: A :class:`~matchcake.operations.state_preparation.ProductState`, a
            :class:`~pennylane.BasisState`, one bit or a sequence of ``k`` bits, or amplitudes of shape
            ``(k, 2)`` or ``(B, k, 2)``.
        :type ancilla: Any
        :return: The product state of the ``k`` ancilla wires.
        :rtype: ProductState
        :raises ValueError: If a scalar or one-dimensional ancilla is not made of the bits 0 and 1.
        """
        if isinstance(ancilla, qml.BasisState):
            ancilla = ProductState.from_basis_state(ancilla)
        if isinstance(ancilla, ProductState):
            return ProductState(ancilla.data[0], wires=range(len(ancilla.wires)), validate_norm=False)
        if qml.math.ndim(ancilla) <= 1:
            bits = np.atleast_1d(np.asarray(ancilla))
            if not np.all((bits == 0) | (bits == 1)):
                raise ValueError(
                    f"A scalar or one-dimensional ancilla is read as bits and must be made of 0 and 1, got "
                    f"{bits.tolist()}. Pass the amplitudes of a product state with shape (k, 2) or (B, k, 2)."
                )
            return ProductState.from_basis_state(bits.astype(int), wires=range(len(bits)))
        return ProductState(ancilla, wires=range(tuple(qml.math.shape(ancilla))[-2]))

    @classmethod
    def from_polarizations(
        cls,
        polarizations: TensorLike,
        sptm: Optional[TensorLike] = None,
        wires: Optional[WiresLike] = None,
        **kwargs,
    ) -> "MixedGaussianState":
        r"""
        Build the state whose normal modes have the given polarizations.

        The covariance matrix is :math:`R^\top \Lambda_D R` with :math:`\Lambda_D` the block-diagonal
        matrix of :func:`~matchcake.utils.covariance.block_diagonal_covariance` and :math:`R` the given
        transition matrix (identity when omitted).

        :param polarizations: Polarizations :math:`\lambda_j \in [-1, 1]` of shape ``(m,)`` or ``(B, m)``.
        :type polarizations: TensorLike
        :param sptm: Single-particle transition matrix of shape ``(2m, 2m)`` or ``(B, 2m, 2m)``. Defaults to
            ``None`` (identity).
        :type sptm: Optional[TensorLike]
        :param wires: Labels of the wires. Defaults to ``range(m)``.
        :type wires: Optional[WiresLike]
        :param kwargs: Forwarded to the constructor.
        :return: The state.
        :rtype: MixedGaussianState
        :raises ValueError: If a polarization is outside of :math:`[-1, 1]`.
        """
        largest = float(np.max(np.abs(np.asarray(detach(polarizations)))))
        if largest > 1.0 + cls.PHYSICALITY_ATOL:
            raise ValueError(
                f"The polarizations of a state lie in [-1, 1], got one of magnitude {largest}: a mode cannot "
                f"be more polarized than a pure mode."
            )
        covariance = block_diagonal_covariance(polarizations)
        if sptm is not None:
            sptm = convert_and_cast_like(sptm, covariance)
            covariance = qml.math.einsum("...ij,...ik,...kl->...jl", sptm, covariance, sptm)
        return cls(covariance, wires=wires, **kwargs)

    def __init__(
        self,
        covariance_matrix: TensorLike,
        wires: Optional[WiresLike] = None,
        *,
        displacement: Optional[TensorLike] = None,
        atol: float = DEFAULT_ATOL,
        source_wires: Optional[WiresLike] = None,
    ):
        shape = tuple(qml.math.shape(covariance_matrix))
        if len(shape) not in (2, 3) or shape[-1] != shape[-2] or shape[-1] % 2 != 0:
            raise ValueError(
                f"The covariance matrix must have shape (2m, 2m) or (B, 2m, 2m) with an even size, got {shape}."
            )
        transposed = qml.math.einsum("...ij->...ji", covariance_matrix)
        if not qml.math.allclose(covariance_matrix, -transposed, atol=self.ANTISYMMETRY_ATOL):
            raise ValueError("The covariance matrix must be real antisymmetric.")
        num_wires = shape[-1] // 2
        if wires is None:
            wires = range(num_wires)
        wires = Wires(wires)
        if len(wires) != num_wires:
            raise ValueError(f"Expected {num_wires} wires for a covariance matrix of shape {shape}, got {len(wires)}.")
        labels = wires.tolist()
        if any(not isinstance(label, (int, np.integer)) for label in labels) or labels != list(
            range(labels[0], labels[0] + num_wires)
        ):
            raise ValueError(
                f"The wires of a {type(self).__name__} must be consecutive integers, got {labels}. "
                f"Use the default wires or relabel them."
            )
        if displacement is not None:
            displacement_shape = tuple(qml.math.shape(displacement))
            if displacement_shape != shape[:-1]:
                raise ValueError(
                    f"The displacement must have shape {shape[:-1]} for a covariance matrix of shape {shape}, "
                    f"got {displacement_shape}."
                )
        self._covariance_matrix = covariance_matrix
        self._displacement = displacement
        self._is_displaced = displacement is not None and not qml.math.allclose(
            detach(displacement), 0.0, atol=self.DISPLACEMENT_ATOL
        )
        self._wires = wires
        self._source_wires = wires if source_wires is None else Wires(source_wires)
        self._atol = float(atol)

    def __repr__(self) -> str:
        displaced = ", is_displaced=True" if self.is_displaced else ""
        return (
            f"{type(self).__name__}(num_wires={self.num_wires}, batch_shape={self.batch_shape}, "
            f"num_mixed_modes={self.num_mixed_modes}, num_pure_states={self.num_pure_states}{displaced})"
        )

    def append(self, ancilla: Any, *, wires: Optional[WiresLike] = None) -> "MixedGaussianState":
        r"""
        Add ancilla wires behind the wires of the state.

        This is :meth:`insert` at the index :attr:`num_wires`, as ``list.append`` is ``list.insert`` at the end.

        :param ancilla: The state of the new wires, see :meth:`insert`.
        :type ancilla: Any
        :param wires: Labels of the wires of the new state. Defaults to consecutive integers from the first
            wire of this state.
        :type wires: Optional[WiresLike]
        :return: The state on the ``m + k`` wires.
        :rtype: MixedGaussianState
        """
        return self.insert(self.num_wires, ancilla, wires=wires)

    def density_matrix(self) -> TensorLike:
        r"""
        Dense :math:`2^m \times 2^m` density matrix of the state, from Wick's theorem.

        The operators :math:`c_S = c_{\mu_1} \cdots c_{\mu_{|S|}}` with :math:`\mu_1 < \cdots < \mu_{|S|}`
        form an orthogonal basis of the operator space, so

        .. math::
            \rho = \frac{1}{2^m} \sum_{|S| \text{ even}} i^{|S| / 2}\, \mathrm{Pf}(\Lambda|_S)\, c_S .

        The sum has :math:`2^{2m - 1}` terms: this method is meant for small systems, typically for
        verification. A displaced state adds the products of an odd number of Majorana operators, whose
        expectation values are the Pfaffians of the extended covariance matrix on :math:`S` and the parity index,

        .. math::
            \rho \mathrel{+}= \frac{1}{2^m} \sum_{|S| \text{ odd}} i^{|S|} (-i)^{(|S| + 1) / 2}\,
            \mathrm{Pf}(\tilde\Lambda|_{S \cup \{2m\}})\, c_S .

        :return: Complex matrix of shape ``(2^m, 2^m)`` or ``(B, 2^m, 2^m)``.
        :rtype: TensorLike
        """
        covariance = self.covariance_matrix
        num_wires = self.num_wires
        majoranas = np.asarray(MajoranaGetter(num_wires).majorana_tensor, dtype=complex)  # (2m, 2^m, 2^m)
        dimension = 2**num_wires
        complex_dtype = complex_dtype_name_like(covariance)
        density = qml.math.cast(convert_like_and_cast_to(np.eye(dimension, dtype=complex), covariance), complex_dtype)
        density = density * qml.math.cast(
            convert_like_and_cast_to(np.ones(self.batch_shape + (1, 1)), covariance), complex_dtype
        )
        for size in range(2, 2 * num_wires + 1, 2):
            index_sets = np.array(list(itertools.combinations(range(2 * num_wires), size)), dtype=int)
            operators = np.stack(
                [np.linalg.multi_dot([majoranas[mu] for mu in subset]) for subset in index_sets], axis=0
            )  # (n_terms, 2^m, 2^m)
            pfaffians = sector_pfaffian_features(covariance, index_sets)  # (..., n_terms)
            pfaffians = qml.math.cast(convert_like_and_cast_to(pfaffians, covariance), complex_dtype)
            operators = qml.math.cast(convert_like_and_cast_to(operators, covariance), complex_dtype)
            density = density + (1j ** (size // 2)) * qml.math.einsum("...s,sij->...ij", pfaffians, operators)
        if not self.is_displaced:
            return density / dimension
        extended = self.extended_covariance_matrix
        for size in range(1, 2 * num_wires, 2):
            index_sets = np.array(list(itertools.combinations(range(2 * num_wires), size)), dtype=int)
            operators = np.stack(
                [reduce(np.matmul, [majoranas[mu] for mu in subset]) for subset in index_sets], axis=0
            )  # (n_terms, 2^m, 2^m)
            parity_column = np.full((len(index_sets), 1), 2 * num_wires, dtype=int)
            extended_sets = np.concatenate([index_sets, parity_column], axis=1)  # (n_terms, size + 1)
            pfaffians = sector_pfaffian_features(extended, extended_sets)  # (..., n_terms)
            pfaffians = qml.math.cast(convert_like_and_cast_to(pfaffians, covariance), complex_dtype)
            operators = qml.math.cast(convert_like_and_cast_to(operators, covariance), complex_dtype)
            phase = (1j**size) * ((-1j) ** ((size + 1) // 2))
            density = density + phase * qml.math.einsum("...s,sij->...ij", pfaffians, operators)
        return density / dimension

    def entropy(self, base: Optional[float] = None) -> TensorLike:
        r"""
        Von Neumann entropy :math:`S(\rho) = -\mathrm{Tr}[\rho \ln \rho]` of the state.

        The normal modes are independent, so the entropy is the sum of the binary entropies of their
        occupation probabilities :math:`p_j = (1 + \lambda_j) / 2`. When the state is the partial trace of a
        pure state, this is the entanglement entropy between the kept and the traced-out wires.

        :param base: Base of the logarithm. Defaults to ``None`` (natural logarithm).
        :type base: Optional[float]
        :return: The entropy, of shape ``()`` or ``(B,)``.
        :rtype: TensorLike
        """
        polarizations: Any = self.mode_polarizations
        probabilities = 0.5 * (1.0 + polarizations)
        occupied_term: Any = self._entropy_term(probabilities)
        empty_term: Any = self._entropy_term(1.0 - probabilities)
        entropy = -qml.math.sum(occupied_term + empty_term, axis=-1)
        if base is not None:
            entropy = entropy / np.log(base)
        return entropy

    def ensemble_observable(self, observable: Operator) -> Operator:
        r"""
        Map an observable on the wires of the state to the observable measured on the members of the ensemble.

        Without displacement the members live on the wires of the state and the observable is returned
        unchanged. The members of a displaced state live on :attr:`ensemble_wires`, where the parity wire holds
        the fermion parity of the wires of the state: a Pauli word with an even number of :math:`X` and
        :math:`Y` factors is unchanged, and a word with an odd number of them is multiplied by an :math:`X` on
        the parity wire. Sums are mapped term by term.

        :param observable: A Pauli observable (word, sum, Hamiltonian), a
            :class:`~matchcake.observables.BatchHamiltonian` or a basis-state :class:`~pennylane.Projector` on
            the wires of the state.
        :type observable: Operator
        :return: The observable on :attr:`ensemble_wires`.
        :rtype: Operator
        :raises NotImplementedError: If the state is displaced and the observable is not a sum of Pauli words.
        """
        if not self.is_displaced or isinstance(observable, BasisStateProjector):
            return observable
        if isinstance(observable, BatchHamiltonian):
            return BatchHamiltonian(observable.coeffs, [self.ensemble_observable(term) for term in observable.ops])
        try:
            coefficients, words = observable.terms()
        except TermsUndefinedError:
            return self._parity_dressed_word(observable)
        dressed_words = [self._parity_dressed_word(word) for word in words]
        if all(dressed_word is word for dressed_word, word in zip(dressed_words, words)):
            return observable
        return qml.Hamiltonian(coefficients, dressed_words)

    def ensemble_operations(self) -> List[Operation]:
        r"""
        Operations preparing every member of the pure-state decomposition as one batched state.

        The basis states of all the members (and of all the elements of the batch of the state, when it has
        one) are stacked into a single batched :class:`~matchcake.operations.state_preparation.ProductState`
        of batch size ``B * K``, followed by the transition-matrix operation of :math:`R` repeated to match.
        Prepending them to a circuit on a :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`
        runs that circuit on every member in one batched execution; the output of member ``k`` of batch
        element ``b`` sits at position ``b * K + k`` of the batch axis of the device.

        The operations act on :attr:`ensemble_wires`: the wires of the state, preceded by the parity wire when
        the state is displaced. In that case the device must be created on :attr:`ensemble_wires` and the
        observables mapped with :meth:`ensemble_observable`.

        :return: A batched :class:`~matchcake.operations.state_preparation.ProductState` followed by a
            :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`.
        :rtype: List[Operation]
        """
        ensemble_wires = self.ensemble_wires
        basis = self._ensemble_basis_states().reshape(-1, len(ensemble_wires))  # (B * K, m) or (B * K, m + 1)
        product_state = ProductState.from_basis_state(basis, wires=ensemble_wires)
        sptm = self.sptm
        if len(self.batch_shape) > 0:
            num_members = self.num_pure_states
            sptm = qml.math.reshape(
                qml.math.stack([sptm] * num_members, axis=1),
                (basis.shape[0], 2 * len(ensemble_wires), 2 * len(ensemble_wires)),
            )
        return [product_state, SingleParticleTransitionMatrixOperation(sptm, wires=ensemble_wires)]

    def insert(self, index: int, ancilla: Any, *, wires: Optional[WiresLike] = None) -> "MixedGaussianState":
        r"""
        Add ancilla wires in a product state at a position of the wire list, as ``list.insert`` does.

        The new state is the tensor product of this state with the ancilla, on ``m + k`` wires: the ``k``
        ancilla wires take the positions ``index, ..., index + k - 1`` and the wires of this state that were
        at ``index`` or behind move back by ``k``. It is the reverse of :meth:`partial_trace`: tracing the
        inserted wires out of the result of :meth:`append` gives this state back.

        The tensor product is the one of the qubits. In the Majorana operators of the ``m + k`` wires, the
        operators of the ancilla carry the parity of the wires in front of it and the operators of the
        wires behind it carry the parity of the ancilla, so the covariance matrix and the displacement of
        the new state combine those of the two factors with the expectation values of these parities.

        Not every tensor product is a Gaussian state, and the rule depends on the ancilla.

        * An ancilla in a computational-basis state can be inserted at any index of any state.
        * An ancilla with superposed qubits can be inserted where the wires in front of it are in a pure
          state: at index 0 of any state, at any index of a product state, at the end of a pure state, and
          behind a block of wires that is pure. Behind wires in a mixed state the tensor product is not a
          Gaussian state, with or without displacement, and a ``ValueError`` is raised.

        The state and the ancilla may each be batched, with the same batch size when both are.

        :param index: Position of the first ancilla wire in the wire list of the new state, from ``0`` (in
            front of every wire) to :attr:`num_wires` (behind every wire). A negative index counts from the
            end, as for a list.
        :type index: int
        :param ancilla: The state of the new wires: a
            :class:`~matchcake.operations.state_preparation.ProductState` or a :class:`~pennylane.BasisState`
            (their wire labels are ignored), one bit or a sequence of ``k`` bits, or the amplitudes of a
            product state with shape ``(k, 2)`` or ``(B, k, 2)``.
        :type ancilla: Any
        :param wires: Labels of the wires of the new state, which must be ``m + k`` consecutive integers.
            Defaults to consecutive integers from the first wire of this state.
        :type wires: Optional[WiresLike]
        :return: The state on the ``m + k`` wires, with the settings of this state.
        :rtype: MixedGaussianState
        :raises ValueError: If the index is out of range, or if the ancilla has superposed qubits and the
            wires in front of it are not in a pure state.
        """
        num_wires = self.num_wires
        position = operator.index(index)
        if position < 0:
            position += num_wires
        if not 0 <= position <= num_wires:
            raise ValueError(
                f"The index {index} is out of range for a {type(self).__name__} on {num_wires} wires: use an "
                f"index between {-num_wires} and {num_wires}."
            )
        ancilla_state = self._ancilla_product_state(ancilla)
        num_ancillas = len(ancilla_state.wires)
        is_basis = bool(qml.math.all(ancilla_state.is_basis_state))
        front_is_pure: Any = True
        if not is_basis and position > 0:
            front_is_pure = self.reduced_state(self.wires[:position]).is_pure
        if not bool(np.all(front_is_pure)):
            raise ValueError(
                f"An ancilla with superposed qubits can only be inserted behind wires that are in a pure state. "
                f"The {position} wires in front of index {index} are in a mixed state, so the tensor product "
                f"is not a Gaussian state. Insert the ancilla at index 0, or use an ancilla in a "
                f"computational-basis state, which can be inserted at any index."
            )

        covariance_matrix = self.covariance_matrix
        amplitudes = ancilla_state.data[0]
        real_dtype = infer_real_dtype(covariance_matrix)
        covariance = torch.as_tensor(covariance_matrix, dtype=real_dtype)  # (..., 2m, 2m)
        displacement = torch.as_tensor(self.displacement, dtype=real_dtype, device=covariance.device)  # (..., 2m)

        # Bloch vectors of the ancilla wires, and its Majorana data with the strings of its own wires.
        ancilla_amplitudes = torch.as_tensor(amplitudes).to(covariance.device)  # (..., k, 2)
        coherence = torch.conj(ancilla_amplitudes[..., 0]) * ancilla_amplitudes[..., 1]
        bloch_x = (2.0 * torch.real(coherence)).to(real_dtype)
        bloch_y = (2.0 * torch.imag(coherence)).to(real_dtype)
        bloch_z = (torch.abs(ancilla_amplitudes[..., 0]) ** 2 - torch.abs(ancilla_amplitudes[..., 1]) ** 2).to(
            real_dtype
        )
        ones = torch.ones_like(bloch_z[..., :1])
        strings_in_front = torch.cat([ones, torch.cumprod(bloch_z[..., :-1], dim=-1)], dim=-1)
        reversed_z = torch.flip(bloch_z, dims=[-1])
        strings_behind = torch.flip(torch.cat([ones, torch.cumprod(reversed_z[..., :-1], dim=-1)], dim=-1), dims=[-1])
        ancilla_batch = tuple(bloch_z.shape[:-1])
        ancilla_parity = torch.prod(bloch_z, dim=-1)  # <Z...Z> of the ancilla
        # <b_nu> and i <b_nu P_B>, with b_nu the Majorana operators of the ancilla and P_B its parity.
        ancilla_displacement = torch.stack([strings_in_front * bloch_x, strings_in_front * bloch_y], dim=-1).reshape(
            ancilla_batch + (2 * num_ancillas,)
        )
        ancilla_parity_displacement = torch.stack(
            [strings_behind * bloch_y, -strings_behind * bloch_x], dim=-1
        ).reshape(ancilla_batch + (2 * num_ancillas,))
        ancilla_covariance = torch.as_tensor(ancilla_state.covariance_matrix).to(
            dtype=real_dtype, device=covariance.device
        )  # (..., 2k, 2k)

        front_size = 2 * position
        if is_basis or position == 0:
            # No parity string of the wires in front is needed: the ancilla has no displacement, or it has
            # no wire in front of it.
            front_parity = torch.ones(self.batch_shape, dtype=real_dtype, device=covariance.device)
            front_parity_displacement = covariance.new_zeros(self.batch_shape + (front_size,))
            behind_displacement = displacement[..., front_size:]
        else:
            front_parity, front_parity_displacement, behind_displacement = self._parity_string_correlators(position)

        batch_shape = torch.broadcast_shapes(self.batch_shape, ancilla_batch)
        size = 2 * (num_wires + num_ancillas)
        front = slice(0, front_size)
        middle = slice(front_size, front_size + 2 * num_ancillas)
        behind = slice(front_size + 2 * num_ancillas, size)
        parity_factor = ancilla_parity[..., None, None]
        joint = covariance.new_zeros(batch_shape + (size, size))
        joint[..., front, front] = covariance[..., :front_size, :front_size]
        joint[..., behind, behind] = covariance[..., front_size:, front_size:]
        joint[..., front, behind] = parity_factor * covariance[..., :front_size, front_size:]
        joint[..., behind, front] = parity_factor * covariance[..., front_size:, :front_size]
        joint[..., middle, middle] = ancilla_covariance
        front_block = front_parity_displacement[..., :, None] * ancilla_displacement[..., None, :]  # (..., 2i, 2k)
        joint[..., front, middle] = front_block
        joint[..., middle, front] = -front_block.mT
        behind_block = ancilla_parity_displacement[..., :, None] * behind_displacement[..., None, :]
        joint[..., middle, behind] = behind_block  # (..., 2k, 2(m - i))
        joint[..., behind, middle] = -behind_block.mT

        new_displacement: Optional[TensorLike] = None
        if self._displacement is not None or not is_basis:
            new_displacement = torch.cat(
                [
                    displacement[..., :front_size].expand(batch_shape + (front_size,)),
                    (front_parity[..., None] * ancilla_displacement).expand(batch_shape + (2 * num_ancillas,)),
                    (ancilla_parity[..., None] * displacement[..., front_size:]).expand(
                        batch_shape + (2 * num_wires - front_size,)
                    ),
                ],
                dim=-1,
            )
        is_tensor = isinstance(covariance_matrix, torch.Tensor) or isinstance(amplitudes, torch.Tensor)
        new_covariance: TensorLike = joint
        if not is_tensor:
            new_covariance = convert_and_cast_like(joint, covariance_matrix)
            if new_displacement is not None:
                new_displacement = convert_and_cast_like(new_displacement, covariance_matrix)
        if wires is None:
            first_label = int(self.wires.tolist()[0])
            wires = range(first_label, first_label + num_wires + num_ancillas)
        return self._new(new_covariance, wires=wires, source_wires=None, displacement=new_displacement)

    def partial_trace(self, wires: WiresLike, **kwargs) -> "MixedGaussianState":
        r"""
        Trace out the given wires and return the state of the remaining ones.

        See :meth:`reduced_state` for the semantics; this method takes the wires to remove instead of the
        wires to keep.

        :param wires: Wires to trace out.
        :type wires: WiresLike
        :param kwargs: Forwarded to :meth:`reduced_state`.
        :return: The reduced state.
        :rtype: MixedGaussianState
        """
        traced = Wires(wires)
        missing = [wire for wire in traced.tolist() if wire not in self.wires]
        if missing:
            raise ValueError(f"Wires {missing} are not wires of this state ({self.wires.tolist()}).")
        kept = [wire for wire in self.wires.tolist() if wire not in traced]
        return self.reduced_state(kept, **kwargs)

    def pure_state_operations(self, index: int) -> List[Operation]:
        r"""
        Operations preparing the member ``index`` of the pure-state decomposition.

        Prepending them to a circuit on a :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`
        with the wires of the state runs that circuit on :math:`V_R |x_\text{index}\rangle`. For a displaced
        state the operations and the device are on :attr:`ensemble_wires`, see :meth:`ensemble_operations`.

        :param index: Index of the member, in ``range(num_pure_states)``.
        :type index: int
        :return: A :class:`~matchcake.operations.state_preparation.ProductState` followed by the
            :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation` of :math:`R`.
        :rtype: List[Operation]
        """
        bits = self._ensemble_basis_states()[..., index, :]
        return [ProductState.from_basis_state(bits, wires=self.ensemble_wires), self.sptm_operation]

    def pure_states(self) -> Iterator[Tuple[TensorLike, ProductState, SingleParticleTransitionMatrixOperation]]:
        r"""
        Iterate over the members of the pure-state decomposition.

        :return: Triples ``(weight, product_state, sptm_operation)``: the weight :math:`w_x` (of shape ``()``
            or ``(B,)``), the basis-state preparation :math:`|x\rangle` and the transition-matrix operation
            :math:`R`, such that the member is :math:`V_R |x\rangle`. For a displaced state both operations
            act on :attr:`ensemble_wires`.
        :rtype: Iterator[Tuple[TensorLike, ProductState, SingleParticleTransitionMatrixOperation]]
        """
        sptm_operation = self.sptm_operation
        weights: Any = self.weights
        for index in range(self.num_pure_states):
            product_state, _ = self.pure_state_operations(index)
            yield weights[..., index], product_state, sptm_operation

    def reduced_state(
        self, wires: WiresLike, *, reduced_wires: Optional[WiresLike] = None, fermionic: bool = False
    ) -> "MixedGaussianState":
        r"""
        Keep the given wires and trace out the others.

        The kept wires are ordered as in the state, and the reduced state is relabeled on ``reduced_wires``
        (``range(k)`` by default). The reduced covariance matrix is the principal submatrix of the covariance
        matrix on the Majorana indices of the kept wires, and the displacement is restricted to the same
        indices. This is the fermionic partial trace: the Jordan-Wigner strings of the reduced system run
        through the kept wires only.

        The fermionic partial trace is the qubit partial trace when no string through a traced-out wire
        survives: when the kept wires form a contiguous block and, for a displaced state, when that block
        starts at the first wire (an operator with an odd number of :math:`X` and :math:`Y` factors keeps
        its string from the first wire). In the other cases the two differ, and the reduction must be
        requested explicitly with ``fermionic=True``. Its result is the qubit partial trace of the state
        whose traced-out wires were first moved behind the kept ones with fSWAP gates, both groups keeping
        their order. In the original state, a Pauli word of the reduced system corresponds to the same word
        multiplied by a :math:`Z` on every traced-out wire that has an odd number of the :math:`X` and
        :math:`Y` factors of the word behind it.

        :param wires: Wires to keep.
        :type wires: WiresLike
        :param reduced_wires: Labels of the wires of the reduced state. Defaults to ``range(k)``.
        :type reduced_wires: Optional[WiresLike]
        :param fermionic: Whether to allow a reduction that differs from the qubit partial trace. Defaults to
            ``False``.
        :type fermionic: bool
        :return: The reduced state.
        :rtype: MixedGaussianState
        :raises ValueError: If a wire is not a wire of the state, if no wire is kept, or if the reduction
            differs from the qubit partial trace and ``fermionic`` is ``False``.
        """
        kept = Wires(wires)
        missing = [wire for wire in kept.tolist() if wire not in self.wires]
        if missing:
            raise ValueError(f"Wires {missing} are not wires of this state ({self.wires.tolist()}).")
        if len(kept) == 0:
            raise ValueError("At least one wire must be kept.")
        positions = sorted(self.wires.indices(kept))
        is_contiguous = positions == list(range(positions[0], positions[0] + len(positions)))
        if not fermionic and not (is_contiguous and (positions[0] == 0 or not self.is_displaced)):
            reason = (
                "the state has a displacement and wires in front of the kept block are traced out"
                if is_contiguous
                else "the kept wires are not a contiguous block"
            )
            raise ValueError(
                f"Keeping the wires at positions {positions} of this {type(self).__name__} is not a qubit partial "
                f"trace: {reason}. Pass fermionic=True to take the fermionic partial trace, which is the qubit "
                f"partial trace of the state whose traced-out wires were first moved behind the kept wires with "
                f"fSWAP gates."
            )
        indices = self.majorana_indices(positions)
        covariance_matrix: Any = self.covariance_matrix
        covariance = covariance_matrix[..., indices[:, None], indices[None, :]]
        source_wires = [self.source_wires.tolist()[position] for position in positions]
        displacement = None
        if self._displacement is not None:
            displacement_vector: Any = self._displacement
            displacement = displacement_vector[..., indices]
        return self._new(covariance, wires=reduced_wires, source_wires=source_wires, displacement=displacement)

    def state_prep(self, wires: Optional[WiresLike] = None) -> GaussianStatePrep:
        r"""
        Return the operation that prepares this state on a device.

        The operation is the first operation of a circuit on a
        :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`, inside or outside a QNode. Called
        inside a QNode, it is queued like any other operation:

        .. code-block:: python

            @qml.qnode(qml.device("nif.qubit", wires=state.num_wires))
            def circuit():
                state.state_prep()
                ...
                return qml.expval(observable)

        :param wires: The wires of the device that hold the state, in the order of the wires of the state.
            Defaults to the wires of the state.
        :type wires: Optional[WiresLike]
        :return: The state preparation.
        :rtype: GaussianStatePrep
        """
        return GaussianStatePrep(
            self.covariance_matrix,
            self._displacement,
            wires=self.wires if wires is None else wires,
            validate=False,
        )

    def _ensemble_basis_states(self) -> np.ndarray:
        """Basis states of the members of the pure-state decomposition, on :attr:`ensemble_wires`.

        The parity wire of a displaced state starts in the zero state for every member: its two values
        prepare states with the same expectation values on the wires of the state.

        :return: Integer array of shape ``(K, m)`` or ``(B, K, m)``, with ``m + 1`` instead of ``m`` for a
            displaced state.
        :rtype: np.ndarray
        """
        basis = self.basis_states
        if not self.is_displaced:
            return basis
        parity_bits = np.zeros(basis.shape[:-1] + (1,), dtype=int)
        return np.concatenate([parity_bits, basis], axis=-1)

    def _new(
        self,
        covariance_matrix: TensorLike,
        wires: Optional[WiresLike],
        source_wires: Optional[WiresLike],
        displacement: Optional[TensorLike] = None,
    ) -> "MixedGaussianState":
        """Build a state sharing the settings of this one.

        :param covariance_matrix: Covariance matrix of the new state.
        :type covariance_matrix: TensorLike
        :param wires: Wires of the new state.
        :type wires: Optional[WiresLike]
        :param source_wires: Source wires of the new state.
        :type source_wires: Optional[WiresLike]
        :param displacement: Displacement vector of the new state. Defaults to ``None`` (no displacement).
        :type displacement: Optional[TensorLike]
        :return: The new state.
        :rtype: MixedGaussianState
        """
        return type(self)(
            covariance_matrix,
            wires=wires,
            displacement=displacement,
            atol=self.atol,
            source_wires=source_wires,
        )

    def _parity_dressed_word(self, word: Operator) -> Operator:
        """Multiply a Pauli word with an odd number of X and Y factors by an X on the parity wire.

        :param word: A Pauli word on the wires of the state.
        :type word: Operator
        :return: The word itself when it has an even number of X and Y factors, its product with an X on the
            parity wire otherwise.
        :rtype: Operator
        :raises NotImplementedError: If the operator is not a Pauli word on the wires of the state.
        """
        wire_map = {wire: position for position, wire in enumerate(self.wires.tolist())}
        try:
            pauli_string = pauli_word_to_string(word, wire_map=wire_map)
        except Exception as error:
            raise NotImplementedError(
                f"The expectation value of {word} cannot be computed on the pure-state decomposition of a "
                f"displaced {type(self).__name__}. Use a Pauli observable, a basis-state Projector or a "
                f"BatchHamiltonian on the wires of the state."
            ) from error
        if (pauli_string.count("X") + pauli_string.count("Y")) % 2 == 0:
            return word
        return qml.prod(qml.PauliX(self.parity_wire), word)

    def _parity_string_correlators(self, position: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Expectation values of the Majorana operators dressed with the parity of the first wires.

        With :math:`F` the Majorana indices of the ``position`` first wires and :math:`P` their parity, the
        three results are :math:`\langle P \rangle`, the vector :math:`i \langle c_\mu P \rangle` for
        :math:`\mu \in F` and the vector :math:`\langle P c_\mu \rangle` for the other indices. Since
        :math:`P = (-i)^{|F|/2} \prod_{\nu \in F} c_\nu`, each of them is a product of Majorana operators and
        is read from the extended covariance matrix with Wick's theorem, :math:`\infty` being its parity index:

        .. math::
            \langle P \rangle = (-1)^{|F|/2}\, \mathrm{Pf}(\Lambda|_F),
            \qquad
            i \langle c_\mu P \rangle
            = (-1)^{|F|/2 + \mu + 1}\, \mathrm{Pf}(\tilde\Lambda|_{F \setminus \mu\, \cup\, \infty}),
            \qquad
            \langle P c_\mu \rangle = (-1)^{|F|/2}\, \mathrm{Pf}(\tilde\Lambda|_{F\, \cup\, \mu\, \cup\, \infty}) .

        :meth:`insert` uses them for an ancilla with a displacement.

        :param position: Number of wires in front, between ``1`` and :attr:`num_wires`.
        :type position: int
        :return: The parity, of shape ``(...)``, and the two vectors, of shapes ``(..., 2 * position)`` and
            ``(..., 2 * (m - position))``.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        num_majoranas = 2 * self.num_wires
        front_size = 2 * position
        extended = self.extended_covariance_matrix
        real_dtype = infer_real_dtype(extended)
        sign = (-1.0) ** position
        front = np.arange(front_size)
        front_parity = (
            sign * torch.as_tensor(sector_pfaffian_features(extended, front[None, :]), dtype=real_dtype)[..., 0]
        )
        without_one = np.stack([np.append(np.delete(front, mu), num_majoranas) for mu in range(front_size)])
        alternating = torch.as_tensor(-sign * (-1.0) ** front, dtype=real_dtype)
        front_terms = alternating * torch.as_tensor(sector_pfaffian_features(extended, without_one), dtype=real_dtype)
        behind_shape = tuple(front_parity.shape) + (num_majoranas - front_size,)
        behind_terms = front_parity.new_zeros(behind_shape)
        if front_size < num_majoranas:
            with_one = np.stack(
                [np.concatenate([front, [mu, num_majoranas]]) for mu in range(front_size, num_majoranas)]
            )
            behind_terms = sign * torch.as_tensor(sector_pfaffian_features(extended, with_one), dtype=real_dtype)
        return front_parity, front_terms, behind_terms

    def _mixed_combinations(self) -> np.ndarray:
        """Every assignment of the bits of the mixed modes, in lexicographic order.

        :return: Integer array of shape ``(K, r)``.
        :rtype: np.ndarray
        """
        num_mixed = self.num_mixed_modes
        return np.array(list(itertools.product([0, 1], repeat=num_mixed)), dtype=int).reshape(2**num_mixed, num_mixed)

    @property
    def atol(self) -> float:
        """The purity tolerance: a mode with :math:`|\\lambda_j| \\geq 1 - \\text{atol}` is treated as pure.

        It is fixed at construction because the purity classification is cached.

        :return: The tolerance.
        :rtype: float
        """
        return self._atol

    @cached_property
    def basis_states(self) -> np.ndarray:
        r"""
        Basis states :math:`x` of the members of the pure-state decomposition.

        Pure modes carry the bit :math:`x_j = 0` when :math:`\lambda_j > 0` and :math:`x_j = 1` otherwise,
        while the mixed modes run over every combination, in lexicographic order.

        :return: Integer array of shape ``(K, m)`` or ``(B, K, m)``.
        :rtype: np.ndarray
        """
        polarizations = np.asarray(detach(self.mode_polarizations))
        pure_bits = (polarizations < 0).astype(int)  # (..., m)
        mixed_modes = self.mixed_modes
        combinations = self._mixed_combinations()
        basis = np.broadcast_to(pure_bits[..., None, :], self.batch_shape + combinations.shape[:1] + (self.num_wires,))
        basis = np.array(basis, dtype=int)
        basis[..., :, mixed_modes] = combinations
        return basis

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """The batch shape of the covariance matrix, ``()`` or ``(B,)``.

        :return: The batch shape.
        :rtype: Tuple[int, ...]
        """
        return tuple(qml.math.shape(self._covariance_matrix))[:-2]

    @property
    def covariance_matrix(self) -> TensorLike:
        """The Majorana covariance matrix of the state, of shape ``(..., 2m, 2m)``.

        :return: The covariance matrix.
        :rtype: TensorLike
        """
        return self._covariance_matrix

    @property
    def displacement(self) -> TensorLike:
        r"""The displacement vector :math:`d_\mu = \mathrm{Tr}[\rho\, c_\mu]` of the state, of shape ``(..., 2m)``.

        It vanishes for a state with a definite fermion parity.

        :return: The displacement vector.
        :rtype: TensorLike
        """
        if self._displacement is None:
            covariance = self.covariance_matrix
            return convert_and_cast_like(np.zeros(self.batch_shape + (2 * self.num_wires,)), covariance)
        return self._displacement

    @property
    def ensemble_wires(self) -> Wires:
        """The wires on which the members of the pure-state decomposition are prepared.

        They are the wires of the state, preceded by :attr:`parity_wire` when the state is displaced.

        :return: The wires of the members.
        :rtype: Wires
        """
        if not self.is_displaced:
            return self.wires
        return Wires([self.parity_wire] + self.wires.tolist())

    @cached_property
    def extended_covariance_matrix(self) -> TensorLike:
        """The covariance matrix bordered by the displacement, of shape ``(..., 2m+1, 2m+1)``.

        The last index is the parity index of the m-Pfaffian strategy. Its row and column vanish for a state
        with a definite parity.

        :return: The extended covariance matrix.
        :rtype: TensorLike
        """
        return extended_covariance_matrix(self.covariance_matrix, self.displacement)

    @property
    def is_displaced(self) -> bool:
        """Whether the state has a nonzero displacement vector, within :attr:`DISPLACEMENT_ATOL`.

        :return: ``True`` for a displaced state.
        :rtype: bool
        """
        return self._is_displaced

    @property
    def is_pure(self) -> TensorLike:
        """Whether every mode is pure, within ``atol``.

        :return: A boolean, or a boolean array of shape ``(B,)``.
        :rtype: TensorLike
        """
        polarizations = np.asarray(detach(self.mode_polarizations))
        return np.all(np.abs(polarizations) >= 1.0 - self.atol, axis=-1)

    @cached_property
    def mixed_modes(self) -> List[int]:
        """Indices of the normal modes that are mixed in at least one element of the batch.

        :return: Sorted mode indices.
        :rtype: List[int]
        """
        polarizations = np.asarray(detach(self.mode_polarizations)).reshape(-1, self.num_wires)
        is_mixed = np.any(np.abs(polarizations) < 1.0 - self.atol, axis=0)
        return [int(index) for index in np.flatnonzero(is_mixed)]

    @property
    def mode_polarizations(self) -> TensorLike:
        r"""
        Polarizations :math:`\lambda_j = \langle Z_j \rangle \in [-1, 1]` of the normal modes.

        :return: Polarizations of shape ``(m,)`` or ``(B, m)``.
        :rtype: TensorLike
        """
        return self._decomposition[1]

    @property
    def num_mixed_modes(self) -> int:
        """The number of mixed normal modes.

        :return: The number of mixed modes.
        :rtype: int
        """
        return len(self.mixed_modes)

    @property
    def num_pure_states(self) -> int:
        """The number :math:`K = 2^r` of members of the pure-state decomposition.

        :return: The number of members.
        :rtype: int
        """
        return 2**self.num_mixed_modes

    @property
    def num_wires(self) -> int:
        """The number of wires of the state.

        :return: The number of wires.
        :rtype: int
        """
        return len(self._wires)

    @property
    def parity_wire(self) -> Optional[int]:
        """The label of the wire that holds the fermion parity in the members of a displaced state.

        It sits in front of the wires of the state, so its label is the first wire label minus one.

        :return: The label, or ``None`` when the state is not displaced.
        :rtype: Optional[int]
        """
        if not self.is_displaced:
            return None
        return int(self.wires.tolist()[0]) - 1

    @property
    def purity(self) -> TensorLike:
        r"""
        Purity :math:`\mathrm{Tr}[\rho^2] = \prod_j \tfrac{1 + \lambda_j^2}{2}` of the state.

        :return: The purity, of shape ``()`` or ``(B,)``.
        :rtype: TensorLike
        """
        polarizations: Any = self.mode_polarizations
        return qml.math.prod(0.5 * (1.0 + polarizations**2), axis=-1)

    @property
    def source_wires(self) -> Wires:
        """The labels, in the original system, of the wires this state was reduced to.

        :return: The source wires.
        :rtype: Wires
        """
        return self._source_wires

    @property
    def sptm(self) -> TensorLike:
        r"""
        Single-particle transition matrix :math:`R \in SO(2m)` of the normal-mode decomposition.

        For a displaced state it is the transition matrix on :attr:`ensemble_wires`, in :math:`SO(2m + 2)`:
        the rotation :math:`\tilde R \in SO(2m + 1)` of the extended covariance matrix, completed by the
        identity and written in the Majorana operators of the :math:`m + 1` wires. With the parity wire in
        front, these are :math:`X_p`, :math:`Y_p` and :math:`Z_p c_\mu`, and the parity index of the extended
        covariance matrix is carried by :math:`Y_p`.

        :return: Matrix of shape ``(2m, 2m)`` or ``(B, 2m, 2m)``, with ``2m + 2`` instead of ``2m`` for a
            displaced state.
        :rtype: TensorLike
        """
        transition_matrix: Any = self._decomposition[0]
        if not self.is_displaced:
            return transition_matrix
        lifted: Any = sptm_lift(transition_matrix)  # (..., 2m+2, 2m+2), parity index 2m and a spare index 2m+1
        num_majoranas = 2 * self.num_wires
        order = np.array([num_majoranas + 1, num_majoranas] + list(range(num_majoranas)), dtype=int)
        return lifted[..., order[:, None], order[None, :]]

    @property
    def sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        """The transition matrix :math:`R` as an operation on :attr:`ensemble_wires`.

        :return: The operation.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        return SingleParticleTransitionMatrixOperation(self.sptm, wires=self.ensemble_wires)

    @cached_property
    def weights(self) -> TensorLike:
        r"""
        Weights :math:`w_x` of the members of the pure-state decomposition, summing to one.

        A pure mode contributes a factor one (its polarization is treated as exactly :math:`\pm 1`), and
        each mixed mode contributes :math:`(1 + (-1)^{x_j} \lambda_j) / 2`. The weights are differentiable
        with respect to the covariance matrix through the polarizations.

        :return: Weights of shape ``(K,)`` or ``(B, K)``.
        :rtype: TensorLike
        """
        polarizations: Any = self.mode_polarizations
        mixed_modes = self.mixed_modes
        combinations = self._mixed_combinations()  # (K, r)
        if len(mixed_modes) == 0:
            return convert_and_cast_like(np.ones(self.batch_shape + (1,)), polarizations)
        signs = convert_and_cast_like(1 - 2 * combinations, polarizations)  # (K, r)
        mixed_polarizations = polarizations[..., mixed_modes]  # (..., r)
        factors = 0.5 * (1.0 + signs * mixed_polarizations[..., None, :])  # (..., K, r)
        return qml.math.prod(factors, axis=-1)

    @property
    def wires(self) -> Wires:
        """The wires of the state.

        :return: The wires.
        :rtype: Wires
        """
        return self._wires

    @cached_property
    def _decomposition(self) -> Tuple[TensorLike, TensorLike]:
        """The block-diagonalization ``(R, polarizations)`` of the covariance matrix.

        A displaced state is decomposed through its extended covariance matrix, and ``R`` then has shape
        ``(..., 2m+1, 2m+1)``.

        :return: The transition matrix and the polarizations.
        :rtype: Tuple[TensorLike, TensorLike]
        :raises ValueError: If a polarization exceeds one, which no state has.
        """
        matrix = self.extended_covariance_matrix if self.is_displaced else self.covariance_matrix
        transition_matrix, polarizations = block_diagonalize_covariance(matrix)
        largest = float(np.max(np.abs(np.asarray(detach(polarizations)))))
        if largest > 1.0 + self.PHYSICALITY_ATOL:
            raise ValueError(
                f"The covariance matrix and the displacement do not describe a state: a normal mode has a "
                f"polarization of magnitude {largest}, larger than one."
            )
        return transition_matrix, polarizations
