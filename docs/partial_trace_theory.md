# Tracing out wires and continuing on a mixed state

This document explains how MatchCake traces out wires of a matchgate circuit, why the
reduced state stays classically tractable although it is mixed, and how a follow-up
circuit is run on it. It builds on the covariance-matrix formalism of
[Computing expectation values using the Covariance-Matrix](expectation_values_theory.md):
the state of the device is a computational-basis state $`|x\rangle`$ evolved by the
matchgate circuit $`V`$, and it is fully described by its Majorana covariance matrix
$`\Lambda = R^\top \Lambda_0 R`$, with $`R`$ the single-particle transition matrix of $`V`$.
Sections 1 to 4 treat this case, Section 5 extends it to a product state with superposed
qubits as the initial state, and Section 6 adds ancilla wires to a state.

## 1. The reduced state is Gaussian

Split the $`n`$ wires into a kept set $`A`$ and a traced-out set $`B`$. The reduced
state $`\rho_A = \mathrm{Tr}_B |\psi\rangle\langle\psi|`$ is characterized by the
expectation values of the operators supported on $`A`$. In the fermionic picture these
are the products of the Majorana operators $`c_\mu`$ with $`\mu \in I_A = \{2a, 2a+1 : a \in A\}`$,
and their expectation values in $`\rho_A`$ are the same as in $`|\psi\rangle`$. Wick's theorem
therefore still applies, with the principal submatrix of the covariance matrix,

```math
\langle c_{\mu_1} \cdots c_{\mu_{2k}} \rangle_{\rho_A}
= i^{-k}\, \mathrm{Pf}\big(\Lambda_A|_{\{\mu_1, \ldots, \mu_{2k}\}}\big),
\qquad
\Lambda_A = \Lambda|_{I_A \times I_A},
```

and the odd products vanish because $`|x\rangle`$ has a definite fermion parity. The
reduced state is thus a fermionic Gaussian state with covariance matrix $`\Lambda_A`$.
Unlike $`\Lambda`$, the submatrix $`\Lambda_A`$ does not square to $`-\mathbb{1}`$ in
general: the reduced state is mixed.

**Qubits versus modes.** The Majorana operators of the kept wires, relabeled as a
system of $`|A|`$ wires, are $`c'_{2j} = Z'_{<j} X'_j`$ and $`c'_{2j+1} = Z'_{<j} Y'_j`$,
whose Jordan-Wigner strings run through the kept wires only. A Pauli word on the reduced
system is a product of the $`c'_\mu`$, and the reduction above identifies $`c'_\mu`$ with the
original $`c_{\mu'}`$ of the corresponding kept wire. In the original system this Pauli word
is therefore dressed with a $`Z`$ on every traced-out wire lying between two of its $`X`$ or $`Y`$
factors, which is the fermionic (mode) partial trace. When $`A`$ is a contiguous block of wires
no traced-out wire lies between two kept wires, and the reduction coincides with the usual
qubit partial trace. For a non-contiguous $`A`$ the two differ, and MatchCake implements the
fermionic one, which keeps the reduced state Gaussian and lets nearest-neighbour matchgates
of the reduced system act as free-fermion hoppings between the kept modes. The fermionic
partial trace has a simple meaning in terms of qubits: it is the qubit partial trace of the
state whose traced-out wires were first moved behind the kept ones with fSWAP gates, both
groups keeping their order. An fSWAP exchanges two neighbouring fermionic modes, so it permutes
the Majorana indices and leaves the principal submatrix of the kept modes unchanged.

## 2. Normal modes and the pure-state decomposition

Every real antisymmetric matrix can be brought to a block-diagonal form by a special
orthogonal matrix. Writing the decomposition in the convention of the single-particle
transition matrices,

```math
\Lambda_A = R_A^\top \Lambda_D R_A,
\qquad
\Lambda_D = \bigoplus_{j=1}^{|A|} (-\lambda_j) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},
\qquad
R_A \in SO(2|A|),
```

the numbers $`\lambda_j \in [-1, 1]`$ are the polarizations $`\langle Z_j \rangle`$ of the
**normal modes**. The block-diagonal matrix $`\Lambda_D`$ is the covariance matrix of a
product of single-mode mixtures, and $`R_A`$ is the transition matrix of a Gaussian unitary
$`V_A`$, so that

```math
\rho_A = V_A \Big( \bigotimes_j \rho_j \Big) V_A^\dagger,
\qquad
\rho_j = \frac{1 + \lambda_j}{2}\, |0\rangle\langle 0| + \frac{1 - \lambda_j}{2}\, |1\rangle\langle 1| .
```

Expanding the product over the computational basis gives the pure-state decomposition

```math
\rho_A = \sum_{x \in \{0, 1\}^{|A|}} w_x\, V_A |x\rangle\langle x| V_A^\dagger,
\qquad
w_x = \prod_j \frac{1 + (-1)^{x_j} \lambda_j}{2} .
```

A mode with $`|\lambda_j| = 1`$ is pure and fixes its bit, so the sum effectively runs over
$`2^r`$ members, $`r`$ being the number of mixed modes. Since the nonzero part of the spectrum
of $`\rho_A`$ equals that of $`\rho_B`$, the number of mixed modes never exceeds
$`\min(|A|, |B|)`$: tracing out a few wires produces a mixture of a few pure states.
Each member $`V_A |x\rangle`$ is a state the device prepares natively, with a
`ProductState` for $`|x\rangle`$ followed by a `SingleParticleTransitionMatrixOperation`
for $`R_A`$.

The entanglement entropy of the cut and the purity follow from the polarizations alone,
with $`p_j = (1 + \lambda_j) / 2`$,

```math
S(\rho_A) = -\sum_j \big[ p_j \ln p_j + (1 - p_j) \ln (1 - p_j) \big],
\qquad
\mathrm{Tr}[\rho_A^2] = \prod_j \frac{1 + \lambda_j^2}{2} .
```

**Computing the decomposition.** The Hermitian matrix $`i\Lambda_A`$ has the eigenvalues
$`\pm\lambda_j`$. For an eigenvector $`v_j`$ of eigenvalue $`\lambda_j > 0`$, the real
vectors $`u_j = \sqrt{2}\,\mathrm{Re}(v_j)`$ and $`w_j = \sqrt{2}\,\mathrm{Im}(v_j)`$ satisfy
$`\Lambda_A u_j = \lambda_j w_j`$ and $`\Lambda_A w_j = -\lambda_j u_j`$, and they form the
rows $`2j`$ and $`2j+1`$ of $`R_A`$. Zero eigenvalues are degenerate between the two
sectors, so their vectors are replaced by an orthonormal basis of the kernel of $`\Lambda_A`$.
A sign flip of one row enforces $`\det R_A = 1`$, and the polarizations are read from the
product $`R_A \Lambda_A R_A^\top`$ so that the result is insensitive to the signs chosen by
the eigensolver.

## 3. Running a follow-up circuit

Let $`U`$ be a matchgate circuit on the reduced system with transition matrix $`R_2`$, and
$`O`$ an observable on it. By linearity of the trace,

```math
\mathrm{Tr}\big[ O\, U \rho_A U^\dagger \big]
= \sum_x w_x\, \langle x | V_A^\dagger U^\dagger\, O\, U V_A | x \rangle ,
```

so the expectation value is the weighted average of the expectation values of the members,
each of which is an ordinary matchgate-circuit simulation starting from the basis state
$`|x\rangle`$ with the global transition matrix $`R_A R_2`$. The members can be run one after the
other or, since the device supports batched state preparations, all at once as a batch of
$`2^r`$ basis states sharing the same transition matrix. Probabilities of computational-basis
outcomes are recombined the same way, and samples of the mixture are drawn by first picking
a member according to the weights $`w_x`$ and then sampling that member.

**Direct evaluation.** The same quantities can be read from the mixed state without
enumerating its members. The covariance matrix evolves linearly, $`\Lambda_A \mapsto
R_2^\top \Lambda_A R_2`$, and Wick's theorem holds for mixed Gaussian states, so the
Pfaffian formulas of the expectation values apply to the evolved covariance matrix. Likewise,
the probability of an outcome $`y`$ on the measured wires $`W`$ is the overlap of two Gaussian
states,

```math
p(y) = \frac{1}{2^{|W|}} \Big| \mathrm{Pf}\big( \Lambda_A|_W + \Lambda_y \big) \Big| ,
```

with $`\Lambda_y`$ the block-diagonal covariance matrix of $`|y\rangle`$. This route costs a
single execution instead of $`2^r`$ and is also the natural way to chain reductions: tracing out
further wires of the evolved mixed state is again a principal submatrix.

## 4. Gradients

The evolved covariance matrix $`\Lambda_A`$ is a differentiable function of the parameters
of the first circuit, and the decomposition is made differentiable by a custom backward pass.
Writing a perturbation of the transition matrix as $`dR_A = \Omega R_A`$ with $`\Omega`$
antisymmetric, the perturbation of the covariance matrix reads

```math
R_A\, d\Lambda_A\, R_A^\top = [\Lambda_D, \Omega] + d\Lambda_D ,
```

which is inverted block by block: the diagonal $`2 \times 2`$ blocks give $`d\lambda_j`$, and
an off-diagonal block $`(j, k)`$ gives the four components of $`\Omega_{jk}`$ divided by
$`\lambda_j - \lambda_k`$ or $`\lambda_j + \lambda_k`$. The directions with a vanishing
denominator (rotations inside a mode, and rotations mixing two modes of equal or opposite
polarization) do not change $`\Lambda_A`$: they are gauge freedoms of the decomposition and
are projected out, which is exact for any loss that only depends on the reduced state. Pure
modes are always degenerate, $`\lambda_j = \pm 1`$, but physical perturbations cannot push a
polarization beyond $`1`$, so only the well-defined components appear and the gradient of the
ensemble is exact there. At an exact degeneracy between two *mixed* modes the decomposition
itself is not differentiable; the direct evaluation of Section 3, which never builds the
decomposition, has a well-defined gradient at every point.

## 5. Product states with superposed qubits

The previous sections start from a computational-basis state, which has a definite fermion
parity. A product state $`\bigotimes_k (a_k|0\rangle + b_k|1\rangle)`$ with superposed qubits
does not: the products of an odd number of Majorana operators no longer vanish, and the state
is described by the extended covariance matrix of
[the expectation-value page](expectation_values_theory.md#5-arbitrary-product-state),

```math
\widetilde\Lambda = \begin{pmatrix} \Lambda & d \\ -d^\top & 0 \end{pmatrix},
\qquad
d_\mu = \langle c_\mu \rangle,
```

whose last index $`2n`$ is the parity index. The state is Gaussian in the algebra of the
$`2n + 1`$ anticommuting operators $`c'_\mu = i\,c_\mu c_\text{par}`$ and
$`c'_{2n} = c_\text{par}`$. A product of an even number of the $`c_\mu`$ is, up to a sign, the
same product of the $`c'_\mu`$, and a product of an odd number of them is, up to a phase, that
product times $`c'_{2n}`$. Both are evaluated by Wick's theorem with $`\widetilde\Lambda`$.

**The reduced state.** The operators supported on the kept modes are still the products of the
$`c_\mu`$ with $`\mu \in I_A`$. An even product is a Pfaffian of $`\Lambda_A`$ as in Section 1,
and an odd product is a Pfaffian of $`\widetilde\Lambda`$ on its indices and the parity index.
Both only involve the principal submatrix of $`\widetilde\Lambda`$ on $`I_A \cup \{2n\}`$,

```math
\widetilde\Lambda_A = \begin{pmatrix} \Lambda_A & d_A \\ -d_A^\top & 0 \end{pmatrix},
\qquad
d_A = d|_{I_A} .
```

Reducing the state therefore restricts the displacement to the kept indices, and the reduced
state is the state of $`|A|`$ wires whose expectation values are given by the readout rule of
the extended covariance matrix applied to $`\widetilde\Lambda_A`$. We call it a *displaced*
Gaussian state. It is generally mixed.

**Qubits versus modes.** The Majorana operator of a kept wire carries a $`Z`$ on every wire in
front of it, traced out or not. For a contiguous block $`A`$, the strings of the wires in front
of the block cancel in a product of an even number of Majorana operators, and one of them
survives in a product of an odd number. With a displacement, the reduction therefore coincides
with the qubit partial trace when $`A`$ is a contiguous block that starts at the first wire,
that is when the last wires are traced out. When a wire in front of $`A`$ is traced out, a Pauli
word of the reduced system with an odd number of $`X`$ and $`Y`$ factors corresponds in the
original system to the same word times a $`Z`$ on that wire. The words with an even number of
$`X`$ and $`Y`$ factors and the probabilities of the computational-basis outcomes are still
those of the qubit partial trace. MatchCake issues a warning in that case.

**Normal modes.** An antisymmetric matrix of odd size has a kernel vector, and its normal form
is

```math
\widetilde\Lambda_A = \widetilde R_A^\top \big( \Lambda_D \oplus 0 \big) \widetilde R_A,
\qquad
\widetilde R_A \in SO(2|A| + 1),
```

with the kernel vector as the last row of $`\widetilde R_A`$. A rotation of the $`2|A| + 1`$
operators $`c'`$ is implemented by a unitary $`\widetilde V_A`$ generated by the quadratic terms
$`c_\mu c_\nu`$ and by the linear terms $`c_\mu`$, the latter being quadratic in the $`c'`$. As
in Section 2,

```math
\rho_A = \widetilde V_A \Big( \bigotimes_j \rho_j \Big) \widetilde V_A^\dagger,
```

so the polarizations $`\lambda_j`$ of $`\widetilde\Lambda_A`$ give the weights $`w_x`$, the
entropy and the purity by the same formulas. The polarizations of $`\Lambda_A`$ alone would be
those of $`\tfrac{1}{2}(\rho_A + P \rho_A P)`$, with $`P`$ the parity of the kept wires, which
is more mixed than $`\rho_A`$.

**The members live on one more wire.** The unitary $`\widetilde V_A`$ is not a matchgate
circuit on the wires of $`A`$, because of its linear generators. Add a *parity wire* $`p`$ in
front of them. The Majorana operators of the $`|A| + 1`$ wires are $`X_p`$, $`Y_p`$ and
$`Z_p c_\mu`$, and the map

```math
T \mapsto T \;\; (T \text{ even}),
\qquad
T \mapsto X_p T \;\; (T \text{ odd})
```

preserves the products of the operators on $`A`$. It sends $`c_\mu`$ to
$`X_p c_\mu = -i\, Y_p (Z_p c_\mu)`$, a product of two Majorana operators of the larger system.
There, $`\widetilde V_A`$ is an ordinary Gaussian unitary: its transition matrix is
$`\widetilde R_A \oplus 1`$, where $`Y_p`$ carries the parity index and $`X_p`$ is the spare
operator. The pure-state decomposition of Section 2 holds on the $`|A| + 1`$ wires, with the
members $`|0\rangle_p |x\rangle`$ evolved by this transition matrix and the same weights
$`w_x`$. A follow-up circuit acts on the wires of $`A`$ only, an even observable is measured as
it is, and an odd observable is measured together with an $`X`$ on the parity wire.

**Direct evaluation.** A follow-up circuit maps
$`\widetilde\Lambda_A \mapsto (R_2 \oplus 1)^\top \widetilde\Lambda_A (R_2 \oplus 1)`$, and the
expectation values are read from it with the readout rule of the extended covariance matrix.
The projectors on the computational-basis outcomes are even, so the probabilities and the
samples only involve $`\Lambda_A`$ and the formula of Section 3 is unchanged.

**Gradients.** The backward pass of Section 4 applies to the odd size with one more family of
gauge directions, the rotations between a mode of zero polarization and the kernel vector.

## 6. Adding ancilla wires

The reverse operation adds wires to a state. Let $`\rho`$ be a state of $`m`$ wires with
covariance matrix $`\Lambda`$ and displacement $`d`$, and let $`\sigma`$ be an ancilla of $`k`$
wires in a product state, inserted at position $`i`$ of the wire list. Call $`F`$ the $`i`$
wires of $`\rho`$ in front of the ancilla, $`B`$ the wires of the ancilla and $`G`$ the wires
of $`\rho`$ behind it. In the register $`(F, B, G)`$ the Jordan-Wigner strings run through the
ancilla, and the Majorana operators are

```math
C_\mu = c_\mu \;\; (\mu \in I_F),
\qquad
C_\nu = P_F\, b_\nu \;\; (\nu \in I_B),
\qquad
C_\mu = c_\mu P_B \;\; (\mu \in I_G),
```

with $`c_\mu`$ the Majorana operators of the wires of $`\rho`$, $`b_\nu`$ those of the ancilla
alone, and $`P_F`$, $`P_B`$ the parities of the wires of $`F`$ and of the ancilla. Since the
tensor product is the one of the qubits, the expectation values factorize, and the covariance
matrix $`i \langle C C \rangle`$ and the displacement $`\langle C \rangle`$ of the new state are

```math
\begin{aligned}
\Lambda'_{FF} &= \Lambda_{FF}, \qquad \Lambda'_{GG} = \Lambda_{GG}, \qquad \Lambda'_{BB} = \Lambda^{\sigma},
\\ \Lambda'_{FG} &= \langle P_B \rangle_\sigma\, \Lambda_{FG},
\\ \Lambda'_{\mu\nu} &= i \langle c_\mu P_F \rangle_\rho\, \langle b_\nu \rangle_\sigma \qquad (\mu \in I_F,\ \nu \in I_B),
\\ \Lambda'_{\nu\mu} &= i \langle b_\nu P_B \rangle_\sigma\, \langle P_F\, c_\mu \rangle_\rho \qquad (\nu \in I_B,\ \mu \in I_G),
\\ d' &= \big( d_F,\ \langle P_F \rangle_\rho\, d^{\sigma},\ \langle P_B \rangle_\sigma\, d_G \big) .
\end{aligned}
```

The quantities of $`\rho`$ are expectation values of Pauli words, read from its extended
covariance matrix: $`\langle P_F \rangle`$ is a string of $`Z`$ on the wires of $`F`$,
$`c_\mu P_F`$ is such a string ending with a $`Y`$ or an $`X`$, and $`P_F c_\mu`$ is a string
of $`Z`$ from the first wire of $`G`$ ending with an $`X`$ or a $`Y`$. For a product state with
Bloch vectors $`(x_l, y_l, z_l)`$, the quantities of the ancilla are

```math
\langle P_B \rangle = \prod_l z_l,
\qquad
i \langle b_{2l} P_B \rangle = y_l \prod_{l' > l} z_{l'},
\qquad
i \langle b_{2l+1} P_B \rangle = -x_l \prod_{l' > l} z_{l'},
```

next to the displacement $`d^\sigma`$ and the covariance matrix $`\Lambda^\sigma`$ of
[the expectation-value page](expectation_values_theory.md#5-arbitrary-product-state).

**When the tensor product is Gaussian.** These formulas always give the two-point functions and
the displacement of $`\rho \otimes \sigma`$. They describe the whole state only when it is a
(displaced) Gaussian state, and this depends on the ancilla.

- *An ancilla in a computational-basis state* $`|x\rangle`$ has no displacement and a definite
  parity $`\langle P_B \rangle = (-1)^{|x|}`$. The only effect on $`\rho`$ is this sign on the
  Majorana operators of $`G`$, and the result is Gaussian for every state and every position.
- *An ancilla with superposed qubits* gives a Gaussian state exactly when the wires of $`F`$
  are in a pure state. In that case $`\rho`$ is the product of a pure state of $`F`$ and a state
  of $`G`$, the ancilla extends the pure state of $`F`$, and the result is again the reduced
  state of a matchgate circuit on a product state (Section 5). This covers position $`0`$ of
  any state, every position of a product state, and the end of a pure state. When the wires
  of $`F`$ are in a mixed state, the reduced state on $`(F, B)`$ is $`\rho_F \otimes \sigma`$,
  which violates Wick's theorem: for one wire with Bloch vector $`r`$ in front of one ancilla
  wire with Bloch vector $`s`$, $`\langle C_0 C_1 C_2 \rangle = i\, s_x`$ while the Pfaffian of
  the extended covariance matrix gives $`i\, |r|^2 s_x`$.

MatchCake computes the new state in the two exact cases and raises an error in the other one.
An ancilla with superposed qubits that is needed next to wires in the middle of a mixed state
can be inserted at position $`0`$ and moved with fSWAP gates. The result differs from the
tensor product by the controlled-$`Z`$ part of these gates, which is a physical interaction
between the ancilla and the wires it crosses.

## 7. Limitations

- The initial state of the first circuit must be a product state. For a product state with
  superposed qubits, the reduction coincides with the qubit partial trace when the kept wires
  are a contiguous block that starts at the first wire. Otherwise the operators with an odd
  number of $`X`$ and $`Y`$ factors are dressed with a $`Z`$ on the traced-out wires in front
  of them (Section 5).
- The reduction is the fermionic partial trace. For a computational-basis initial state it
  equals the qubit partial trace for a contiguous block of kept wires, and differs from it by
  Jordan-Wigner strings through the traced-out wires otherwise.
- An ancilla with superposed qubits can only be added behind wires that are in a pure state
  (Section 6). An ancilla in a computational-basis state can be added anywhere.
- The number of members of the pure-state decomposition grows as $`2^r`$ with the number
  $`r`$ of mixed modes, at most the number of traced-out wires. The direct evaluation does
  not have this cost.

## References

[1] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics." arXiv preprint
quant-ph/0404180 (2004).

[2] Peschel, Ingo. "Calculation of reduced density matrices from correlation functions."
Journal of Physics A: Mathematical and General 36, no. 14 (2003): L205.

[3] Brod, Daniel J. "Efficient classical simulation of matchgate circuits with generalized
inputs and measurements." Physical Review A 93, no. 6 (2016): 062332.
