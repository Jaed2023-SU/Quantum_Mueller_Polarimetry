"""
eapt_psi_plus_cp_chi_mle.py

CP-only quantum process tomography from polarization-entangled photon-pair
coincidence counts, setting default input Bell state is

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
from scipy.optimize import least_squares  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import os


STATE_ORDER = ["H", "V", "D", "A", "R", "L"]
PAULI_ORDER = ["I", "X", "Y", "Z"]

class Quantum_EAPT_NTP:
    """
    CP-only non-trace-preserving EAPT/QPT analysis class.

    This class contains the original script logic above the "# Example" block.
    The numerical routines are static methods, so they can be called as
    `Quantum_EAPT_NTP.method(...)` or from an instance.
    """

    STATE_ORDER = STATE_ORDER
    PAULI_ORDER = PAULI_ORDER
    
    
    # -----------------------------------------------------------------------------
    # Polarization states and projectors
    # -----------------------------------------------------------------------------
    @staticmethod
    def header_from_raw(input_csv, output_csv):
        label_map = {1: "H", 2: "V", 3: "D", 4: "A", 5: "R", 6: "L"}
    
        df = pd.read_csv(input_csv, header=None)
        df.columns = ["Signal", "Idler", "count"]
    
        df["Signal"] = df["Signal"].astype(int).map(label_map)
        df["Idler"] = df["Idler"].astype(int).map(label_map)
        df["count"] = df["count"].astype(int)
    
        df.to_csv(output_csv, index=False)
        return df
    
    @staticmethod
    def _find_column_case_insensitive(df: pd.DataFrame, name: str) -> str:
        """Return the actual DataFrame column name matching `name` case-insensitively."""
        norm = {str(c).strip().lower(): c for c in df.columns}
        key = name.strip().lower()
        if key not in norm:
            raise ValueError(
                f"Column {name!r} was not found. Available columns are {list(df.columns)!r}."
            )
        return norm[key]
    
    
    @staticmethod
    def load_labeled_coincidence_csv(
        input_csv: str,
        reference_col: str = "Idler",
        channel_col: str = "Signal",
        count_col: str = "count",
        states: list[str] | tuple[str, ...] = STATE_ORDER,
        output_matrix_csv: str | None = None,
    ) -> np.ndarray:
        """Load a labeled entangled-photon coincidence CSV as a 6x6 count matrix.
    
        Expected CSV format, matching the uploaded example:
    
            Signal,Idler,count
            H,H,44
            H,V,3052
            ...
    
        The returned matrix convention is the one used by the EAPT likelihood:
    
            rows    = reference/idler projection basis  [H,V,D,A,R,L]
            columns = channel/signal projection basis   [H,V,D,A,R,L]
    
        Therefore, by default, this function maps
    
            matrix[Idler, Signal] = count.
    
        If your experiment uses the opposite naming convention, swap
        `reference_col` and `channel_col`.
        """
        df = pd.read_csv(input_csv)
    
        ref_name = Quantum_EAPT_NTP._find_column_case_insensitive(df, reference_col)
        ch_name = Quantum_EAPT_NTP._find_column_case_insensitive(df, channel_col)
        cnt_name = Quantum_EAPT_NTP._find_column_case_insensitive(df, count_col)
    
        state_list = [str(s).upper() for s in states]
        state_to_index = {s: i for i, s in enumerate(state_list)}
    
        counts = np.full((len(state_list), len(state_list)), np.nan, dtype=float)
    
        for row_number, row in df.iterrows():
            ref = str(row[ref_name]).strip().upper()
            ch = str(row[ch_name]).strip().upper()
            if ref not in state_to_index:
                if isinstance(row_number, (int, np.integer)):
                    row_display = str(int(row_number) + 2)
                else:
                    row_display = str(row_number)
                raise ValueError(
                    f"Unknown reference basis {ref!r} at CSV row {row_display}. "
                    f"Allowed bases are {state_list}."
                )
            if ch not in state_to_index:
                if isinstance(row_number, (int, np.integer)):
                    row_display = str(int(row_number) + 2)
                else:
                    row_display = str(row_number)
                raise ValueError(
                    f"Unknown channel basis {ch!r} at CSV row {row_display}. "
                    f"Allowed bases are {state_list}."
                )
    
            i = state_to_index[ref]
            j = state_to_index[ch]
            if np.isfinite(counts[i, j]):
                raise ValueError(
                    f"Duplicate coincidence entry for reference={ref}, channel={ch}."
                )
            counts[i, j] = float(pd.to_numeric(row[cnt_name], errors="raise"))
    
        missing = []
        for i, ref in enumerate(state_list):
            for j, ch in enumerate(state_list):
                if not np.isfinite(counts[i, j]):
                    missing.append(f"{ref},{ch}")
        if missing:
            raise ValueError(
                "The labeled CSV does not contain all 36 reference/channel basis pairs. "
                f"Missing entries: {missing}"
            )
        if np.any(counts < 0):
            raise ValueError("Coincidence counts must be non-negative.")
    
        if output_matrix_csv is not None:
            pd.DataFrame(counts, index=state_list, columns=state_list).to_csv(output_matrix_csv)
    
        return counts
    
    
    @staticmethod
    def third_col_to_matrix(input_csv, output_csv):
        """Backward-compatible helper for old headerless files.
    
        New labeled files should use load_labeled_coincidence_csv instead.
        """
        df = pd.read_csv(input_csv, header=None)
        data = pd.to_numeric(df.iloc[:, 2], errors="raise").to_numpy()
    
        if len(data) != 36:
            raise ValueError("The third column must have exactly 36 values.")
    
        matrix = data.reshape(6, 6)
        pd.DataFrame(matrix).to_csv(output_csv, index=False, header=False)
        return matrix
    
    @staticmethod
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
    
    
    @staticmethod
    def projector(label: str) -> np.ndarray:
        v = Quantum_EAPT_NTP.ket(label)
        return np.outer(v, v.conj())
    
    
    @staticmethod
    def projector_list(states=STATE_ORDER) -> list[np.ndarray]:
        return [Quantum_EAPT_NTP.projector(s) for s in states]
    
    
    @staticmethod
    def bell_state(name = "Psi+") -> np.ndarray:
        """Return Bell state in basis [HH, HV, VH, VV].
    
        The first qubit is the reference arm; the second qubit is the channel arm.
        """
        H = Quantum_EAPT_NTP.ket("H")
        V = Quantum_EAPT_NTP.ket("V")
        HH = np.kron(H, H)
        HV = np.kron(H, V)
        VH = np.kron(V, H)
        VV = np.kron(V, V)
        
        name = name.replace(" ", "").lower()
    
        if name in ["phi+", "phiplus", "phi_plus"]:
    
            state = (HH + VV) / np.sqrt(2)
    
        elif name in ["phi-", "phiminus", "phi_minus"]:
    
            state = (HH - VV) / np.sqrt(2)
    
        elif name in ["psi+", "psiplus", "psi_plus"]:
    
            state = (HV + VH) / np.sqrt(2)
    
        elif name in ["psi-", "psiminus", "psi_minus"]:
    
            state = (HV - VH) / np.sqrt(2)
    
        else:
    
            raise ValueError(
    
                "Unknown Bell state. Use 'Phi+', 'Phi-', 'Psi+', or 'Psi-'."
            )
            
        return np.outer(state, state.conj())
    
    
    # -----------------------------------------------------------------------------
    # Cholesky parameterization for a 4x4 positive semidefinite Choi matrix
    # -----------------------------------------------------------------------------
    
    
    @staticmethod
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
    
    
    @staticmethod
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
    
    
    @staticmethod
    def initial_identity_choi_t(mixing: float = 1e-4) -> np.ndarray:
        """Initial Cholesky parameters close to the identity channel Choi matrix."""
        I2 = np.eye(2, dtype=complex)
        v = I2.reshape(-1, order="F")[:, None]
        J_id = v @ v.conj().T  # trace = 2
        rhoJ = J_id / np.trace(J_id)
        rhoJ = (1.0 - mixing) * rhoJ + mixing * np.eye(4, dtype=complex) / 4.0
        rhoJ = rhoJ / np.trace(rhoJ)
        return Quantum_EAPT_NTP.density_to_t(rhoJ)
    
    
    # -----------------------------------------------------------------------------
    # Channel action, Choi matrix, and chi matrix
    # -----------------------------------------------------------------------------
    
    
    @staticmethod
    def params_to_choi(t: np.ndarray, d: int = 2) -> np.ndarray:
        """Return CP Choi matrix J=d*rhoJ with fixed trace d, but no TP constraint."""
        rhoJ = Quantum_EAPT_NTP.t_to_density(t, dim=d * d)
        J = d * rhoJ
        return 0.5 * (J + J.conj().T)
    
    
    @staticmethod
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
    
    
    @staticmethod
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
    
    
    @staticmethod
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
    
    
    @staticmethod
    def choi_trace_preserving_error(J: np.ndarray) -> np.ndarray:
        """Diagnostic only: Tr_output(J)-I. Not used as a fitting constraint."""
        return Quantum_EAPT_NTP.partial_trace_output_from_choi(J) - np.eye(2, dtype=complex)
    
    
    @staticmethod
    def trace_nonincreasing_matrix(J: np.ndarray) -> np.ndarray:
        """Return I - Tr_output(J). PSD means trace-nonincreasing.
    
        This is diagnostic only. The fit enforces CP, not trace preservation and not
        trace non-increase. Because Tr(J)=2 is used as a gauge, this diagnostic may
        have positive and negative eigenvalues for some data.
        """
        return np.eye(2, dtype=complex) - Quantum_EAPT_NTP.partial_trace_output_from_choi(J)
    
    
    @staticmethod
    def pauli_stokes_basis():
        sigma0 = np.array([[1, 0], [0, 1]], dtype = complex)
        sigma1 = np.array([[0, 1], [1, 0]], dtype = complex)
        sigma2 = np.array([[0, -1j], [1j, 0]], dtype = complex)
        sigma3 = np.array([[1, 0], [0, -1]], dtype = complex)

        Pauli = [sigma0, sigma1, sigma2, sigma3]
        Stokes = [sigma0, sigma3, sigma1, sigma2]

        Pauli_label = ["I", "X", "Y", "Z"]
        Stokes_label = ["$S_0$", "$S_1$", "$S_2$", "$S_3$"]
        return Pauli, Stokes, Pauli_label, Stokes_label

    @staticmethod
    def kraus_to_chi(kraus: list[np.ndarray]) -> np.ndarray:
        """Return process matrix chi in unnormalized Pauli basis {I, X, Y, Z}.
    
        E(rho) = sum_{m,n} chi[m,n] P_m rho P_n^dagger.
        Since Tr(P_m^dagger P_n)=2 delta_mn,
            a_{mu,m} = Tr(P_m^dagger K_mu) / 2.
        """
        P = Quantum_EAPT_NTP.pauli_stokes_basis()[0]
        chi = np.zeros((4, 4), dtype=complex)
        for K in kraus:
            a = np.array([np.trace(Pm.conj().T @ K) / 2.0 for Pm in P], dtype=complex)
            chi += np.outer(a, a.conj())
        return 0.5 * (chi + chi.conj().T)
    
    
    @staticmethod
    def chi_tp_matrix(chi: np.ndarray) -> np.ndarray:
        """Return sum_mn chi_mn P_n^dagger P_m. Equals I only for TP maps."""
        P = Quantum_EAPT_NTP.pauli_stokes_basis()[0]
        accum = np.zeros((2, 2), dtype=complex)
        for m in range(4):
            for n in range(4):
                accum += chi[m, n] * (P[n].conj().T @ P[m])
        return accum
    
    
    # -----------------------------------------------------------------------------
    # MLE prediction and residual
    # -----------------------------------------------------------------------------
    
    
    @staticmethod
    def predict_coincidences_from_choi(
        J: np.ndarray,
        intensity: float,
        accidentals: np.ndarray | float = 0.0,
        states = STATE_ORDER,
        bell_state_name: str = "Psi+"
    ) -> np.ndarray:
        """Predict a 6x6 coincidence matrix for Psi+ input and a CP channel."""
        kraus = Quantum_EAPT_NTP.choi_to_kraus(J)
        rho_in = Quantum_EAPT_NTP.bell_state(name = bell_state_name)
        rho_out = Quantum_EAPT_NTP.apply_channel_second_qubit(rho_in, kraus)
        projs = Quantum_EAPT_NTP.projector_list(states)
    
        pred = np.zeros((len(states), len(states)), dtype=float)
        for i, Pref in enumerate(projs):
            for j, Pch in enumerate(projs):
                M = np.kron(Pref, Pch)
                p = np.real(np.trace(M @ rho_out))
                pred[i, j] = max(p, 0.0) * intensity
    
        pred = pred + np.asarray(accidentals, dtype=float)
        return np.maximum(pred, 0.01)
    
    
    @staticmethod
    def _unpack_fit_params(x: np.ndarray) -> tuple[np.ndarray, float]:
        t = x[:16]
        log_intensity = x[16]
        J = Quantum_EAPT_NTP.params_to_choi(t, d=2)
        intensity = float(np.exp(log_intensity))
        return J, intensity
    
    
    @staticmethod
    def _mle_residual(
        x: np.ndarray,
        counts: np.ndarray,
        accidentals: np.ndarray,
        weights: np.ndarray | None,
        bell_state_name: str = "Psi+"
    ) -> np.ndarray:
        J, intensity = Quantum_EAPT_NTP._unpack_fit_params(x)
        pred = Quantum_EAPT_NTP.predict_coincidences_from_choi(J, intensity, accidentals = accidentals, bell_state_name = bell_state_name)
        residual = (pred - counts) / np.sqrt(pred)
        if weights is not None:
            residual = residual * weights
        return residual.ravel()
    
    
    @staticmethod
    def qpt_mle_chi_from_psi_plus_coincidences_cp_only(
        counts: np.ndarray | str,
        bell_state_name: str = "Psi+",
        accidentals: np.ndarray | float = 0.0,
        max_nfev: int = 20000,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        verbose: int = 0,
        weights: np.ndarray | None = None,
        reference_col: str = "Idler",
        channel_col: str = "Signal",
        count_col: str = "count",
    ) -> dict:
        """Fit a CP-only single-qubit process from 6x6 Psi+ coincidence counts.
    
        Trace preservation is deliberately NOT imposed. The result is guaranteed to
        be completely positive because the Choi matrix is parameterized as
        J = 2*T*T^dagger/Tr(T*T^dagger).  The trace of J is fixed to 2 only as a
        scale gauge; this is not equivalent to trace preservation.
    
        Parameters
        ----------
        counts:
            Either a 6x6 coincidence-count matrix or a labeled CSV path.
            Matrix convention: rows are the reference/idler arm and columns are the
            channel/signal arm, both ordered as [H, V, D, A, R, L].
            For a labeled CSV with columns Signal, Idler, count, the default mapping is
            counts[Idler, Signal] = count.
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
        if isinstance(counts, (str, os.PathLike)):
            counts = Quantum_EAPT_NTP.load_labeled_coincidence_csv(
                str(counts),
                reference_col=reference_col,
                channel_col=channel_col,
                count_col=count_col,
            )
        else:
            counts = np.asarray(counts, dtype=float)
    
        if counts.shape != (6, 6):
            raise ValueError("counts must be a 6x6 matrix with order [H,V,D,A,R,L], or a labeled CSV path.")
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
        init_t = Quantum_EAPT_NTP.initial_identity_choi_t()
        x0 = np.concatenate([init_t, [np.log(init_intensity)]])
    
        result = least_squares(
            Quantum_EAPT_NTP._mle_residual,
            x0,
            args=(counts, accidentals, weights, bell_state_name),
            method="trf",
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=verbose,
        )
    
        J, intensity = Quantum_EAPT_NTP._unpack_fit_params(result.x)
        kraus = Quantum_EAPT_NTP.choi_to_kraus(J)
        predicted = Quantum_EAPT_NTP.predict_coincidences_from_choi(J, intensity, accidentals=accidentals)
        rho_out = Quantum_EAPT_NTP.apply_channel_second_qubit(Quantum_EAPT_NTP.bell_state(name = bell_state_name), kraus)
        chi = Quantum_EAPT_NTP.kraus_to_chi(kraus)
    
        choi_eigs = np.linalg.eigvalsh(J)
        chi_eigs = np.linalg.eigvalsh(chi)
        tp_from_choi = Quantum_EAPT_NTP.choi_trace_preserving_error(J)
        chi_tp = Quantum_EAPT_NTP.chi_tp_matrix(chi) - np.eye(2)
        tni_matrix = Quantum_EAPT_NTP.trace_nonincreasing_matrix(J)
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
    
    @staticmethod
    def plot_matrix(M: np.ndarray, labels: list, title_prefix: str, save_name: str | None = None):
    
        M = np.asarray(M)
        
        for part_name, data in [("Real", M.real), ("Imaginary", M.imag)]:
            fig, ax = plt.subplots(figsize = (5, 4))
            im = ax.imshow(data, vmin = -1, vmax = 1, cmap="coolwarm")
            ax.set_title(f"{title_prefix}: {part_name}")
            ax.set_xticks(range(4), labels)
            ax.set_yticks(range(4), labels)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{data[i, j]:.3f}", ha = "center", va = "center")
            fig.colorbar(im, ax = ax)
            fig.tight_layout()
            if save_name is not None:
                plt.savefig(save_name + f"_{part_name}.png", dpi = 500, bbox_inches = "tight")
    
    
    
    @staticmethod
    def _basis_labels_for_matrix(name: str, shape: tuple[int, ...]) -> tuple[list[str] | None, list[str] | None]:
        """Return row/column labels appropriate for a saved fit-result matrix."""
        if len(shape) != 2:
            return None, None
        if name == "predicted" and shape == (6, 6):
            return Quantum_EAPT_NTP.STATE_ORDER, Quantum_EAPT_NTP.STATE_ORDER
        if shape == (4, 4):
            if name in {"chi", "chi_tp_error_diagnostic"}:
                return Quantum_EAPT_NTP.PAULI_ORDER, Quantum_EAPT_NTP.PAULI_ORDER
            return ["HH", "HV", "VH", "VV"], ["HH", "HV", "VH", "VV"]
        if shape == (2, 2):
            return ["H", "V"], ["H", "V"]
        return None, None
    
    
    @staticmethod
    def _save_real_table(path: str, data: np.ndarray, row_labels=None, col_labels=None) -> None:
        """Save a real-valued matrix with optional row/column labels."""
        arr = np.asarray(data)
        if arr.ndim == 0:
            pd.DataFrame({"value": [float(np.real(arr))]}).to_csv(path, index=False)
            return
        if arr.ndim == 1:
            index = row_labels if row_labels is not None and len(row_labels) == arr.shape[0] else None
            pd.DataFrame({"value": np.real(arr)}, index=index).to_csv(path)
            return
        index = row_labels if row_labels is not None and len(row_labels) == arr.shape[0] else None
        columns = col_labels if col_labels is not None and len(col_labels) == arr.shape[1] else None
        pd.DataFrame(np.real(arr), index=index, columns=columns).to_csv(path)
    
    
    @staticmethod
    def _save_complex_matrix_tables(folder: str, prefix: str, name: str, M: np.ndarray) -> None:
        """Save a complex matrix as labeled real/imag/abs/phase CSV tables."""
        arr = np.asarray(M)
        row_labels, col_labels = Quantum_EAPT_NTP._basis_labels_for_matrix(name, arr.shape)
        safe_name = name.replace("/", "_").replace(" ", "_")
    
        if np.iscomplexobj(arr):
            Quantum_EAPT_NTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_real.csv"), arr.real, row_labels, col_labels)
            Quantum_EAPT_NTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_imag.csv"), arr.imag, row_labels, col_labels)
            Quantum_EAPT_NTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_abs.csv"), np.abs(arr), row_labels, col_labels)
            Quantum_EAPT_NTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_phase_rad.csv"), np.angle(arr), row_labels, col_labels)
        else:
            Quantum_EAPT_NTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}.csv"), arr, row_labels, col_labels)
    
    
    @staticmethod
    def save_fit_return_values_as_csv_tables(fit: dict, folder: str, prefix: str = "fit") -> None:
        """Save qpt_mle_chi_from_psi_plus_coincidences_cp_only return values as CSV tables.
    
        The function writes labeled CSV files into `folder`:
            - chi, J, rho_out: 4x4 complex tables with HH/HV/VH/VV or Pauli labels
            - predicted: 6x6 real table with H,V,D,A,R,L labels
            - kraus: one 2x2 complex table per Kraus operator
            - cp_diagnostics: scalar summary plus matrix-valued diagnostics
            - result: least_squares scalar summary and optimized parameter vector
        """
        os.makedirs(folder, exist_ok=True)
    
        scalar_rows = []
    
        for key, value in fit.items():
            if key == "cp_diagnostics":
                cp_diagnostics_folder = folder + "/cp_diagnostics"
                os.makedirs(cp_diagnostics_folder, exist_ok=True)
                diag_scalar_rows = []
                for dkey, dval in value.items():
                    darr = np.asarray(dval)
                    if darr.ndim == 0:
                        scalar_val = np.asarray(dval).item()
                        diag_scalar_rows.append({
                            "quantity": dkey,
                            "real": float(np.real(scalar_val)),
                            "imag": float(np.imag(scalar_val)),
                        })
                    else:
                        Quantum_EAPT_NTP._save_complex_matrix_tables(cp_diagnostics_folder, prefix, f"cp_diagnostics_{dkey}", darr)
                pd.DataFrame(diag_scalar_rows).to_csv(
                    os.path.join(cp_diagnostics_folder, f"{prefix}_cp_diagnostics_scalars.csv"), index=False
                )
                continue
    
            if key == "kraus":
                for k, K in enumerate(value):
                    kraus_folder = folder + "/Kraus_operators"
                    os.makedirs(kraus_folder, exist_ok=True)
                    Quantum_EAPT_NTP._save_complex_matrix_tables(kraus_folder, prefix, f"kraus_{k:02d}", np.asarray(K))
                continue
    
            if key == "result":
                opt_result_folder = folder + "/optimization_result"
                os.makedirs(opt_result_folder, exist_ok=True)
                result_summary = {
                    "success": bool(value.success),
                    "status": int(value.status),
                    "message": str(value.message),
                    "cost": float(value.cost),
                    "optimality": float(value.optimality),
                    "nfev": int(value.nfev),
                    "njev": int(value.njev) if value.njev is not None else -1,
                }
                pd.DataFrame([result_summary]).to_csv(
                    os.path.join(opt_result_folder, f"{prefix}_least_squares_result_summary.csv"), index=False
                )
                pd.DataFrame({"parameter_index": np.arange(len(value.x)), "value": value.x}).to_csv(
                    os.path.join(opt_result_folder, f"{prefix}_least_squares_optimized_parameters.csv"), index=False
                )
                pd.DataFrame({"residual_index": np.arange(len(value.fun)), "value": value.fun}).to_csv(
                    os.path.join(opt_result_folder, f"{prefix}_least_squares_residuals.csv"), index=False
                )
                continue
    
            arr = np.asarray(value)
            result_folder = folder + "/result"
            os.makedirs(result_folder, exist_ok=True)
            if arr.ndim == 0:
                scalar_val = arr.item()
                scalar_rows.append({
                    "quantity": key,
                    "real": float(np.real(scalar_val)),
                    "imag": float(np.imag(scalar_val)),
                })
            elif arr.ndim in (1, 2):
                Quantum_EAPT_NTP._save_complex_matrix_tables(result_folder, prefix, key, arr)
            else:
                pd.DataFrame({"flat_index": np.arange(arr.size), "value": arr.reshape(-1)}).to_csv(
                    os.path.join(result_folder, f"{prefix}_{key}_flattened.csv"), index=False
                )
    
        if scalar_rows:
            pd.DataFrame(scalar_rows).to_csv(os.path.join(folder, f"{prefix}_scalars.csv"), index=False)
    
    @staticmethod
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
    
    
