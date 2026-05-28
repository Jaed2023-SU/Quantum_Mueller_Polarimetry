"""
eapt_psi_plus_cp_chi_mle.py

CP-only quantum process tomography from polarization-entangled photon-pair
coincidence counts, assuming the input Bell state is

    |Psi+> = (|H,V> + |V,H>) / sqrt(2),

where the FIRST qubit is the reference/idler arm and the SECOND qubit is the
channel/signal arm. The channel is fitted directly by MLE as a completely
positive process, but trace preservation is NOT imposed.

Input data convention
---------------------
counts is a 6x6 matrix with row/column order

    [H, V, D, A, R, L]

    rows    = reference-arm analyzer projection
    columns = channel-arm analyzer projection

Therefore counts[i, j] is the coincidence count measured with reference
projection states[i] and channel projection states[j].

Important normalization note
----------------------------
Without the trace-preserving constraint, the overall scale of the Choi matrix
and the fitted coincidence intensity are not separately identifiable from a
single 6x6 coincidence table.  This code therefore fixes the Choi gauge by
using

    J = d * rhoJ,    Tr(rhoJ)=1,    d=2,

so Tr(J)=2.  This does NOT impose trace preservation.  Trace preservation would
require Tr_output(J)=I, which is deliberately not constrained here.  The fitted
intensity absorbs the overall coincidence scale/loss.

The CP constraint is imposed by the uploaded-code-style lower-triangular
Cholesky parameterization

    rhoJ = T T^dagger / Tr(T T^dagger),
    J    = 2 rhoJ.

Consequently J is Hermitian positive semidefinite throughout the optimization.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


STATE_ORDER = ["H", "V", "D", "A", "R", "L"]
PAULI_ORDER = ["I", "X", "Y", "Z"]


# -----------------------------------------------------------------------------
# Polarization states and projectors
# -----------------------------------------------------------------------------


def ket(label: str) -> np.ndarray:
    """Return a polarization ket in the H/V computational basis.

    Convention used here:
        |R> = (|H> - i |V>) / sqrt(2)
        |L> = (|H> + i |V>) / sqrt(2)

    Swap these two definitions if your optical convention is opposite.
    """
    label = label.upper()
    if label == "H":
        return np.array([1.0, 0.0], dtype=complex)
    if label == "V":
        return np.array([0.0, 1.0], dtype=complex)
    if label == "D":
        return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    if label == "A":
        return np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
    if label == "R":
        return np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2)
    if label == "L":
        return np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)
    raise ValueError(f"Unknown polarization label: {label!r}")


def projector(label: str) -> np.ndarray:
    v = ket(label)
    return np.outer(v, v.conj())


def projector_list(states=STATE_ORDER) -> list[np.ndarray]:
    return [projector(s) for s in states]


def psi_plus_density() -> np.ndarray:
    """Return |Psi+><Psi+| in basis [HH, HV, VH, VV].

    The first qubit is the reference arm; the second qubit is the channel arm.
    """
    H = ket("H")
    V = ket("V")
    psi = (np.kron(H, V) + np.kron(V, H)) / np.sqrt(2)
    return np.outer(psi, psi.conj())


# -----------------------------------------------------------------------------
# Cholesky parameterization for a 4x4 positive semidefinite Choi matrix
# -----------------------------------------------------------------------------


def t_to_density(t: np.ndarray, dim: int = 4) -> np.ndarray:
    """Convert real Cholesky parameters to a trace-one PSD density matrix.

    This follows the same idea used in many quantum-tomography MLE codes:
    an unconstrained real vector defines a complex lower-triangular matrix T,
    and rho = T T^dagger / Tr(T T^dagger).

    Parameter order:
        - first dim entries: real diagonal of T
        - then, for each lower-triangular off-diagonal element T[row, col]
          with row > col, two entries: real part, imaginary part.

    For dim=4, this uses 4 + 2*6 = 16 real parameters.
    """
    t = np.asarray(t, dtype=float)
    expected = dim + dim * (dim - 1)
    if t.size != expected:
        raise ValueError(f"Expected {expected} real parameters for dim={dim}, got {t.size}.")

    T = np.zeros((dim, dim), dtype=complex)
    k = 0
    for i in range(dim):
        T[i, i] = t[k]
        k += 1
    for col in range(dim):
        for row in range(col + 1, dim):
            T[row, col] = t[k] + 1j * t[k + 1]
            k += 2

    rho = T @ T.conj().T
    tr = np.real(np.trace(rho))
    if tr <= 0:
        raise FloatingPointError("Cholesky matrix produced zero trace.")
    rho = rho / tr
    return 0.5 * (rho + rho.conj().T)


def density_to_t(rho: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Construct a Cholesky parameter vector from a positive density matrix."""
    rho = np.asarray(rho, dtype=complex)
    dim = rho.shape[0]
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho + eps * np.eye(dim, dtype=complex)
    rho = rho / np.trace(rho)
    T = np.linalg.cholesky(rho)

    values: list[float] = []
    for i in range(dim):
        values.append(float(np.real(T[i, i])))
    for col in range(dim):
        for row in range(col + 1, dim):
            values.append(float(np.real(T[row, col])))
            values.append(float(np.imag(T[row, col])))
    return np.array(values, dtype=float)


def initial_identity_choi_t(mixing: float = 1e-4) -> np.ndarray:
    """Initial Cholesky parameters close to the identity channel Choi matrix."""
    I2 = np.eye(2, dtype=complex)
    v = I2.reshape(-1, order="F")[:, None]
    J_id = v @ v.conj().T  # trace = 2
    rhoJ = J_id / np.trace(J_id)
    rhoJ = (1.0 - mixing) * rhoJ + mixing * np.eye(4, dtype=complex) / 4.0
    rhoJ = rhoJ / np.trace(rhoJ)
    return density_to_t(rhoJ)


# -----------------------------------------------------------------------------
# Channel action, Choi matrix, and chi matrix
# -----------------------------------------------------------------------------


def params_to_choi(t: np.ndarray, d: int = 2) -> np.ndarray:
    """Return CP Choi matrix J=d*rhoJ with fixed trace d, but no TP constraint."""
    rhoJ = t_to_density(t, dim=d * d)
    J = d * rhoJ
    return 0.5 * (J + J.conj().T)


def choi_to_kraus(J: np.ndarray, tol: float = 1e-12) -> list[np.ndarray]:
    """Convert a PSD Choi matrix to Kraus operators.

    Vectorization convention: column stacking, vec(K)=K.reshape(-1, order='F').
    """
    J = 0.5 * (np.asarray(J, dtype=complex) + np.asarray(J, dtype=complex).conj().T)
    eigvals, eigvecs = np.linalg.eigh(J)
    kraus: list[np.ndarray] = []
    for val, vec in zip(eigvals, eigvecs.T):
        if val > tol:
            K = np.sqrt(val) * vec.reshape((2, 2), order="F")
            kraus.append(K)
    if len(kraus) == 0:
        kraus.append(np.zeros((2, 2), dtype=complex))
    return kraus


def apply_channel_second_qubit(rho_2q: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    """Apply I tensor E to a two-qubit density matrix.

    If E is not trace preserving, the returned rho_out is generally
    unnormalized. This is expected in this CP-only model.
    """
    out = np.zeros_like(rho_2q, dtype=complex)
    I = np.eye(2, dtype=complex)
    for K in kraus:
        op = np.kron(I, K)
        out += op @ rho_2q @ op.conj().T
    return 0.5 * (out + out.conj().T)


def partial_trace_output_from_choi(J: np.ndarray) -> np.ndarray:
    """Compute Tr_output(J) for column-stacked Choi matrix.

    For a trace-preserving channel this equals I. This code does not enforce it.
    """
    out = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for o in range(2):
                out[i, j] += J[o + 2 * i, o + 2 * j]
    return out


def choi_trace_preserving_error(J: np.ndarray) -> np.ndarray:
    """Diagnostic only: Tr_output(J)-I. Not used as a fitting constraint."""
    return partial_trace_output_from_choi(J) - np.eye(2, dtype=complex)


def trace_nonincreasing_matrix(J: np.ndarray) -> np.ndarray:
    """Return I - Tr_output(J). PSD means trace-nonincreasing.

    This is diagnostic only. The fit enforces CP, not trace preservation and not
    trace non-increase. Because Tr(J)=2 is used as a gauge, this diagnostic may
    have positive and negative eigenvalues for some data.
    """
    return np.eye(2, dtype=complex) - partial_trace_output_from_choi(J)


def pauli_matrices() -> list[np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]


def kraus_to_chi(kraus: list[np.ndarray]) -> np.ndarray:
    """Return process matrix chi in unnormalized Pauli basis {I, X, Y, Z}.

    E(rho) = sum_{m,n} chi[m,n] P_m rho P_n^dagger.
    Since Tr(P_m^dagger P_n)=2 delta_mn,
        a_{mu,m} = Tr(P_m^dagger K_mu) / 2.
    """
    P = pauli_matrices()
    chi = np.zeros((4, 4), dtype=complex)
    for K in kraus:
        a = np.array([np.trace(Pm.conj().T @ K) / 2.0 for Pm in P], dtype=complex)
        chi += np.outer(a, a.conj())
    return 0.5 * (chi + chi.conj().T)


def chi_tp_matrix(chi: np.ndarray) -> np.ndarray:
    """Return sum_mn chi_mn P_n^dagger P_m. Equals I only for TP maps."""
    P = pauli_matrices()
    accum = np.zeros((2, 2), dtype=complex)
    for m in range(4):
        for n in range(4):
            accum += chi[m, n] * (P[n].conj().T @ P[m])
    return accum


# -----------------------------------------------------------------------------
# MLE prediction and residual
# -----------------------------------------------------------------------------


def predict_coincidences_from_choi(
    J: np.ndarray,
    intensity: float,
    accidentals: np.ndarray | float = 0.0,
    states=STATE_ORDER,
) -> np.ndarray:
    """Predict a 6x6 coincidence matrix for Psi+ input and a CP channel."""
    kraus = choi_to_kraus(J)
    rho_in = psi_plus_density()
    rho_out = apply_channel_second_qubit(rho_in, kraus)
    projs = projector_list(states)

    pred = np.zeros((len(states), len(states)), dtype=float)
    for i, Pref in enumerate(projs):
        for j, Pch in enumerate(projs):
            M = np.kron(Pref, Pch)
            p = np.real(np.trace(M @ rho_out))
            pred[i, j] = max(p, 0.0) * intensity

    pred = pred + np.asarray(accidentals, dtype=float)
    return np.maximum(pred, 0.01)


def _unpack_fit_params(x: np.ndarray) -> tuple[np.ndarray, float]:
    t = x[:16]
    log_intensity = x[16]
    J = params_to_choi(t, d=2)
    intensity = float(np.exp(log_intensity))
    return J, intensity


def _mle_residual(
    x: np.ndarray,
    counts: np.ndarray,
    accidentals: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    J, intensity = _unpack_fit_params(x)
    pred = predict_coincidences_from_choi(J, intensity, accidentals=accidentals)
    residual = (pred - counts) / np.sqrt(pred)
    if weights is not None:
        residual = residual * weights
    return residual.ravel()


def qpt_mle_chi_from_psi_plus_coincidences_cp_only(
    counts: np.ndarray,
    accidentals: np.ndarray | float = 0.0,
    max_nfev: int = 20000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
    verbose: int = 0,
    weights: np.ndarray | None = None,
) -> dict:
    """Fit a CP-only single-qubit process from 6x6 Psi+ coincidence counts.

    Trace preservation is deliberately NOT imposed. The result is guaranteed to
    be completely positive because the Choi matrix is parameterized as
    J = 2*T*T^dagger/Tr(T*T^dagger).  The trace of J is fixed to 2 only as a
    scale gauge; this is not equivalent to trace preservation.

    Parameters
    ----------
    counts:
        6x6 coincidence-count matrix. Row order and column order are both
        [H, V, D, A, R, L]. Rows are the reference arm, columns are the channel arm.
    accidentals:
        Scalar or 6x6 matrix of accidental coincidences added to the prediction.
        If you already subtracted accidentals from the counts, leave this as 0.
    weights:
        Optional 6x6 multiplicative weights applied to the residuals.

    Returns
    -------
    dict with keys:
        chi, J, kraus, rho_out, predicted, intensity, cp_diagnostics, result
    """
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (6, 6):
        raise ValueError("counts must be a 6x6 matrix with order [H,V,D,A,R,L].")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")

    accidentals = np.asarray(accidentals, dtype=float)
    if accidentals.shape == ():
        accidentals = np.full((6, 6), float(accidentals))
    if accidentals.shape != (6, 6):
        raise ValueError("accidentals must be either a scalar or a 6x6 matrix.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (6, 6):
            raise ValueError("weights must be a 6x6 matrix if provided.")

    init_intensity = max(float(np.nanmax(counts - accidentals)) * 2.0, 1.0)
    init_t = initial_identity_choi_t()
    x0 = np.concatenate([init_t, [np.log(init_intensity)]])

    result = least_squares(
        _mle_residual,
        x0,
        args=(counts, accidentals, weights),
        method="trf",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=verbose,
    )

    J, intensity = _unpack_fit_params(result.x)
    kraus = choi_to_kraus(J)
    predicted = predict_coincidences_from_choi(J, intensity, accidentals=accidentals)
    rho_out = apply_channel_second_qubit(psi_plus_density(), kraus)
    chi = kraus_to_chi(kraus)

    choi_eigs = np.linalg.eigvalsh(J)
    chi_eigs = np.linalg.eigvalsh(chi)
    tp_from_choi = choi_trace_preserving_error(J)
    chi_tp = chi_tp_matrix(chi) - np.eye(2)
    tni_matrix = trace_nonincreasing_matrix(J)
    tni_eigs = np.linalg.eigvalsh(0.5 * (tni_matrix + tni_matrix.conj().T))

    return {
        "chi": chi,
        "J": J,
        "kraus": kraus,
        "rho_out": rho_out,
        "predicted": predicted,
        "intensity": intensity,
        "fval": float(np.sum(result.fun**2)),
        "result": result,
        "cp_diagnostics": {
            "min_choi_eigenvalue": float(np.min(choi_eigs)),
            "min_chi_eigenvalue": float(np.min(chi_eigs)),
            "trace_choi": float(np.real(np.trace(J))),
            "trace_chi": float(np.real(np.trace(chi))),
            "choi_tp_error_diagnostic": tp_from_choi,
            "chi_tp_error_diagnostic": chi_tp,
            "trace_nonincreasing_matrix_I_minus_TrOutJ": tni_matrix,
            "min_trace_nonincreasing_eigenvalue": float(np.min(tni_eigs)),
            "rho_out_trace_for_psi_plus": float(np.real(np.trace(rho_out))),
        },
    }


# Backward-friendly shorter alias
qpt_mle_chi_from_psi_plus_coincidences = qpt_mle_chi_from_psi_plus_coincidences_cp_only


# -----------------------------------------------------------------------------
# Convenience plotting and printing
# -----------------------------------------------------------------------------


def print_complex_matrix(M: np.ndarray, name: str = "matrix", precision: int = 4) -> None:
    """Print a compact complex matrix."""
    M = np.asarray(M)
    print(f"{name} =")
    for row in M:
        entries = []
        for z in row:
            entries.append(f"{z.real:+.{precision}f}{z.imag:+.{precision}f}j")
        print("  [" + ", ".join(entries) + "]")


def plot_chi(chi: np.ndarray, title_prefix: str = "MLE CP-only chi") -> None:
    """Plot real and imaginary parts of chi as annotated heatmaps."""
    chi = np.asarray(chi)
    labels = PAULI_ORDER

    for part_name, data in [("Real", chi.real), ("Imag", chi.imag)]:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(data, vmin=-1, vmax=1)
        ax.set_title(f"{title_prefix}: {part_name}")
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()


def counts_from_flat_36(values: list[float] | np.ndarray) -> np.ndarray:
    """Convert a flat length-36 list into the 6x6 matrix convention.

    The flat order is row-major:
        HH, HV, HD, HA, HR, HL,
        VH, VV, VD, VA, VR, VL,
        ...,
        LH, LV, LD, LA, LR, LL.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size != 36:
        raise ValueError("values must contain exactly 36 numbers.")
    return arr.reshape(6, 6)


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example only. Replace this with your measured coincidence matrix.
    # Row = reference analyzer [H,V,D,A,R,L]
    # Col = channel analyzer   [H,V,D,A,R,L]
    example_counts = np.array(
        [
            [120, 9800, 5000, 5000, 5000, 5000],
            [9700, 140, 5000, 5000, 5000, 5000],
            [5000, 5000, 9900, 100, 5000, 5000],
            [5000, 5000, 130, 9850, 5000, 5000],
            [5000, 5000, 5000, 5000, 9800, 150],
            [5000, 5000, 5000, 5000, 120, 9750],
        ],
        dtype=float,
    )

    fit = qpt_mle_chi_from_psi_plus_coincidences_cp_only(example_counts)
    print_complex_matrix(fit["chi"], "chi")
    print("intensity =", fit["intensity"])
    print("fval =", fit["fval"])
    print("CP diagnostics:")
    for key, val in fit["cp_diagnostics"].items():
        print(key, "=", val)
