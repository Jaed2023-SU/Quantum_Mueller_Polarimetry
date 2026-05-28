"""
eapt_psi_plus_cptp_chi_mle.py

CPTP quantum process tomography from polarization-entangled photon-pair
coincidence counts, assuming the input Bell state is

    |Psi+> = (|H,V> + |V,H>) / sqrt(2),

where the FIRST qubit is the reference/idler arm and the SECOND qubit is the
channel/signal arm.  The channel is fitted directly as a CPTP map by MLE.

Input data convention
---------------------
counts is a 6x6 matrix with row/column order

    [H, V, D, A, R, L]

    rows    = reference-arm analyzer projection
    columns = channel-arm analyzer projection

Therefore counts[i, j] is the coincidence count measured with
reference projection states[i] and channel projection states[j].

This file intentionally follows the style of the uploaded Quantum-Tomography
code: lower-level NumPy arrays, least-squares MLE residuals of the form
(prediction - counts) / sqrt(prediction), and explicit matrix construction.

The CPTP constraint is imposed by parameterizing the channel through Kraus
operators K_mu satisfying

    sum_mu K_mu^dagger K_mu = I.

This is achieved by constructing an isometry V from unconstrained real
parameters via QR decomposition and unstacking V into Kraus operators.
Consequently, complete positivity and trace preservation are hard constraints,
not post-fit corrections.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Polarization states and projectors
# -----------------------------------------------------------------------------

STATE_ORDER = ["H", "V", "D", "A", "R", "L"]
PAULI_ORDER = ["I", "X", "Y", "Z"]


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
# CPTP channel parameterization by Kraus isometry
# -----------------------------------------------------------------------------


def _params_to_complex_matrix(params: np.ndarray, rows: int = 8, cols: int = 2) -> np.ndarray:
    """Convert real vector to a complex matrix A of shape (rows, cols)."""
    params = np.asarray(params, dtype=float)
    need = 2 * rows * cols
    if params.size != need:
        raise ValueError(f"Expected {need} real parameters, got {params.size}.")
    real = params[: rows * cols].reshape(rows, cols)
    imag = params[rows * cols :].reshape(rows, cols)
    return real + 1j * imag


def _complex_matrix_to_params(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=complex)
    return np.concatenate([A.real.ravel(), A.imag.ravel()])


def params_to_kraus(params: np.ndarray, n_kraus: int = 4) -> list[np.ndarray]:
    """Map unconstrained real parameters to CPTP Kraus operators.

    A complex 2*n_kraus by 2 matrix is orthonormalized to an isometry V using
    QR decomposition.  V is then split into n_kraus blocks of size 2x2.

    Since V^dagger V = I, the resulting Kraus operators obey
        sum_mu K_mu^dagger K_mu = I.
    """
    if n_kraus < 1:
        raise ValueError("n_kraus must be >= 1.")
    rows = 2 * n_kraus
    A = _params_to_complex_matrix(params, rows=rows, cols=2)

    # Avoid a rank-deficient starting point.  The tiny diagonal bias is harmless
    # during optimization but stabilizes QR if all parameters are accidentally zero.
    A = A.copy()
    A[0, 0] += 1e-12
    A[1, 1] += 1e-12

    Q, R = np.linalg.qr(A, mode="reduced")

    # Fix arbitrary QR column phases for a more continuous parameterization.
    diag = np.diag(R)
    phase = np.ones_like(diag, dtype=complex)
    nz = np.abs(diag) > 1e-14
    phase[nz] = diag[nz] / np.abs(diag[nz])
    Q = Q * phase.conj()[None, :]

    kraus = [Q[2 * mu : 2 * (mu + 1), :] for mu in range(n_kraus)]
    return kraus


def kraus_tp_error(kraus: list[np.ndarray]) -> np.ndarray:
    accum = np.zeros((2, 2), dtype=complex)
    for K in kraus:
        accum += K.conj().T @ K
    return accum - np.eye(2, dtype=complex)


def initial_identity_params(n_kraus: int = 4, noise: float = 1e-3, seed: int | None = 1) -> np.ndarray:
    """Initial parameters near the identity channel."""
    rng = np.random.default_rng(seed)
    V = np.zeros((2 * n_kraus, 2), dtype=complex)
    V[0:2, :] = np.eye(2, dtype=complex)  # K0 = I, others = 0
    V = V + noise * (rng.normal(size=V.shape) + 1j * rng.normal(size=V.shape))
    return _complex_matrix_to_params(V)


# -----------------------------------------------------------------------------
# Channel action, Choi matrix, and chi matrix
# -----------------------------------------------------------------------------


def apply_channel_second_qubit(rho_2q: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    """Apply I ⊗ E to a two-qubit density matrix."""
    out = np.zeros_like(rho_2q, dtype=complex)
    I = np.eye(2, dtype=complex)
    for K in kraus:
        op = np.kron(I, K)
        out += op @ rho_2q @ op.conj().T
    return 0.5 * (out + out.conj().T)


def kraus_to_choi(kraus: list[np.ndarray]) -> np.ndarray:
    """Return the Choi matrix J_E = sum_mu |K_mu>><<K_mu|.

    Vectorization convention: column stacking, vec(K) = K.reshape(-1, order='F').
    With this convention, trace_out(J_E) = I for a trace-preserving channel.
    """
    J = np.zeros((4, 4), dtype=complex)
    for K in kraus:
        v = K.reshape(-1, order="F")[:, None]
        J += v @ v.conj().T
    return 0.5 * (J + J.conj().T)


def partial_trace_output_from_choi(J: np.ndarray) -> np.ndarray:
    """Compute Tr_output(J) for column-stacked Choi matrix.

    For a 2x2 Kraus operator K, vec(K) with order='F' has index

        composite_index = output + 2 * input.

    Hence Tr_output(J)[i,j] = sum_o J[o + 2*i, o + 2*j].
    For a trace-preserving channel this equals I.
    """
    out = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for o in range(2):
                out[i, j] += J[o + 2 * i, o + 2 * j]
    return out


def choi_trace_preserving_error(J: np.ndarray) -> np.ndarray:
    """Compute Tr_output(J) - I."""
    return partial_trace_output_from_choi(J) - np.eye(2, dtype=complex)


def pauli_matrices() -> list[np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]


def kraus_to_chi(kraus: list[np.ndarray]) -> np.ndarray:
    """Return process matrix chi in unnormalized Pauli basis {I, X, Y, Z}.

    E(rho) = sum_{m,n} chi[m,n] P_m rho P_n^dagger.

    Since Tr(P_m^dagger P_n)=2 delta_mn, coefficients are
        a_{mu,m} = Tr(P_m^dagger K_mu) / 2.
    """
    P = pauli_matrices()
    chi = np.zeros((4, 4), dtype=complex)
    for K in kraus:
        a = np.array([np.trace(Pm.conj().T @ K) / 2.0 for Pm in P], dtype=complex)
        chi += np.outer(a, a.conj())
    return 0.5 * (chi + chi.conj().T)


def chi_tp_matrix(chi: np.ndarray) -> np.ndarray:
    """Return sum_mn chi_mn P_n^dagger P_m, which equals I for TP."""
    P = pauli_matrices()
    accum = np.zeros((2, 2), dtype=complex)
    for m in range(4):
        for n in range(4):
            accum += chi[m, n] * (P[n].conj().T @ P[m])
    return accum


# -----------------------------------------------------------------------------
# MLE prediction and residual
# -----------------------------------------------------------------------------


def predict_coincidences(
    kraus: list[np.ndarray],
    intensity: float,
    accidentals: np.ndarray | float = 0.0,
    states=STATE_ORDER,
) -> np.ndarray:
    """Predict a 6x6 coincidence matrix for Psi+ input and a CPTP channel."""
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


def _unpack_fit_params(x: np.ndarray, n_kraus: int) -> tuple[list[np.ndarray], float]:
    n_channel_params = 2 * (2 * n_kraus) * 2
    kraus_params = x[:n_channel_params]
    log_intensity = x[n_channel_params]
    kraus = params_to_kraus(kraus_params, n_kraus=n_kraus)
    intensity = float(np.exp(log_intensity))
    return kraus, intensity


def _mle_residual(
    x: np.ndarray,
    counts: np.ndarray,
    accidentals: np.ndarray,
    n_kraus: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    kraus, intensity = _unpack_fit_params(x, n_kraus=n_kraus)
    pred = predict_coincidences(kraus, intensity, accidentals=accidentals)
    residual = (pred - counts) / np.sqrt(pred)
    if weights is not None:
        residual = residual * weights
    return residual.ravel()


def qpt_mle_chi_from_psi_plus_coincidences(
    counts: np.ndarray,
    accidentals: np.ndarray | float = 0.0,
    n_kraus: int = 4,
    seed: int | None = 1,
    max_nfev: int = 20000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
    verbose: int = 0,
    weights: np.ndarray | None = None,
) -> dict:
    """Fit a CPTP single-qubit process from 6x6 Psi+ coincidence counts.

    Parameters
    ----------
    counts:
        6x6 coincidence-count matrix. Row order and column order are both
        [H, V, D, A, R, L]. Rows are the reference arm, columns are the channel arm.
    accidentals:
        Scalar or 6x6 matrix of accidental coincidences added to the prediction.
        If you already subtracted accidentals from the counts, leave this as 0.
    n_kraus:
        Number of Kraus operators in the CPTP model. For a general qubit channel,
        n_kraus=4 is sufficient.
    weights:
        Optional 6x6 multiplicative weights applied to the residuals.

    Returns
    -------
    dict with keys:
        chi, J, kraus, rho_out, predicted, intensity, cptp_errors, result
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

    # Coincidence probabilities for 36 projectors do not sum to one, so the
    # intensity is an empirical count scale.  Use a robust starting value.
    init_intensity = max(float(np.nanmax(counts - accidentals)) * 2.0, 1.0)

    init_kraus_params = initial_identity_params(n_kraus=n_kraus, seed=seed)
    x0 = np.concatenate([init_kraus_params, [np.log(init_intensity)]])

    result = least_squares(
        _mle_residual,
        x0,
        args=(counts, accidentals, n_kraus, weights),
        method="trf",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=verbose,
    )

    kraus, intensity = _unpack_fit_params(result.x, n_kraus=n_kraus)
    predicted = predict_coincidences(kraus, intensity, accidentals=accidentals)
    rho_out = apply_channel_second_qubit(psi_plus_density(), kraus)
    J = kraus_to_choi(kraus)
    chi = kraus_to_chi(kraus)

    choi_eigs = np.linalg.eigvalsh(J)
    chi_eigs = np.linalg.eigvalsh(chi)
    tp_from_kraus = kraus_tp_error(kraus)
    tp_from_choi = choi_trace_preserving_error(J)
    chi_tp = chi_tp_matrix(chi) - np.eye(2)

    return {
        "chi": chi,
        "J": J,
        "kraus": kraus,
        "rho_out": rho_out,
        "predicted": predicted,
        "intensity": intensity,
        "fval": float(np.sum(result.fun**2)),
        "result": result,
        "cptp_errors": {
            "min_choi_eigenvalue": float(np.min(choi_eigs)),
            "min_chi_eigenvalue": float(np.min(chi_eigs)),
            "kraus_tp_error": tp_from_kraus,
            "choi_tp_error": tp_from_choi,
            "chi_tp_error": chi_tp,
            "trace_choi": float(np.real(np.trace(J))),
            "trace_chi": float(np.real(np.trace(chi))),
        },
    }


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


def plot_chi(chi: np.ndarray, title_prefix: str = "MLE CPTP chi") -> None:
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

    fit = qpt_mle_chi_from_psi_plus_coincidences(example_counts)
    print_complex_matrix(fit["chi"], "chi")
    print("intensity =", fit["intensity"])
    print("fval =", fit["fval"])
    print("CPTP diagnostics:")
    for key, val in fit["cptp_errors"].items():
        print(key, "=", val)
