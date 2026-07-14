from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import os

    # -----------------------------------------------------------------------------
    # Polarization states and projectors
    # -----------------------------------------------------------------------------

STATE_ORDER = ["H", "V", "D", "A", "R", "L"]
PAULI_ORDER = ["I", "X", "Y", "Z"]

class Quantum_EAPT_CPTP:
    
    STATE_ORDER = STATE_ORDER
    PAULI_ORDER = PAULI_ORDER

    @staticmethod
    def header_from_raw(input_csv, output_csv):
        label_map = {1: "H", 2: "V", 3: "D", 4: "A", 5: "R", 6: "L"}

        df = pd.read_csv(input_csv, header = None)
        df.columns = ["Signal", "Idler", "count"]

        df["Signal"] = df["Signal"].astype(int).map(label_map)
        df["Idler"] = df["Idler"].astype(int).map(label_map)
        df["count"] = df["count"].astype(int)

        df.to_csv(output_csv, index = False)
        return df

    @staticmethod
    def _find_column_case_insensitive(df: pd.DataFrame, name: str):
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
    ):
        
        df = pd.read_csv(input_csv)
    
        ref_name = Quantum_EAPT_CPTP._find_column_case_insensitive(df, reference_col)
        ch_name = Quantum_EAPT_CPTP._find_column_case_insensitive(df, channel_col)
        cnt_name = Quantum_EAPT_CPTP._find_column_case_insensitive(df, count_col)
    
        state_list = [str(s).upper() for s in states]
        state_to_index = {s: i for i, s in enumerate(state_list)}
    
        counts = np.full((len(state_list), len(state_list)), np.nan, dtype = float)
    
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
        df = pd.read_csv(input_csv, header = None)
        data = pd.to_numeric(df.iloc[:, 2], errors = "raise").to_numpy()
    
        if len(data) != 36:
            raise ValueError("The third column must have exactly 36 values.")
    
        matrix = data.reshape(6, 6)
        pd.DataFrame(matrix).to_csv(output_csv, index = False, header = False)
        return matrix

    @staticmethod
    def ket(label: str):
        label = label.upper()
        if label == "H":
            return np.array([1.0, 0.0], dtype = complex)
        if label == "V":
            return np.array([0.0, 1.0], dtype = complex)
        if label == "D":
            return np.array([1.0, 1.0], dtype = complex) / np.sqrt(2)
        if label == "A":
            return np.array([1.0, -1.0], dtype = complex) / np.sqrt(2)
        if label == "R":
            return np.array([1.0, -1.0j], dtype = complex) / np.sqrt(2)
        if label == "L":
            return np.array([1.0, 1.0j], dtype = complex) / np.sqrt(2)
        raise ValueError(f"Unknown polarization label: {label!r}")


    @staticmethod
    def projector(label: str):
        v = Quantum_EAPT_CPTP.ket(label)
        return np.outer(v, v.conj())


    @staticmethod
    def projector_list(states = STATE_ORDER):
        return [Quantum_EAPT_CPTP.projector(s) for s in states]


    @staticmethod
    def bell_state(name = "Psi+"):
            """Return Bell state in basis [HH, HV, VH, VV].
        
            The first qubit is the reference arm; the second qubit is the channel arm.
            """
            H = Quantum_EAPT_CPTP.ket("H")
            V = Quantum_EAPT_CPTP.ket("V")
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
    # CPTP channel parameterization by Kraus isometry
    # -----------------------------------------------------------------------------

    @staticmethod
    def _params_to_complex_matrix(params: np.ndarray, rows: int = 8, cols: int = 2):
        """Convert real vector to a complex matrix A of shape (rows, cols)."""
        params = np.asarray(params, dtype=float)
        need = 2 * rows * cols
        if params.size != need:
            raise ValueError(f"Expected {need} real parameters, got {params.size}.")
        real = params[: rows * cols].reshape(rows, cols)
        imag = params[rows * cols :].reshape(rows, cols)
        return real + 1j * imag


    @staticmethod
    def _complex_matrix_to_params(A: np.ndarray):
        A = np.asarray(A, dtype = complex)
        return np.concatenate([A.real.ravel(), A.imag.ravel()])


    @staticmethod
    def params_to_kraus(params: np.ndarray, n_kraus: int = 4):
        if n_kraus < 1:
            raise ValueError("n_kraus must be >= 1.")
        rows = 2 * n_kraus
        A = Quantum_EAPT_CPTP._params_to_complex_matrix(params, rows=rows, cols=2)

        A = A.copy()
        A[0, 0] += 1e-12
        A[1, 1] += 1e-12

        Q, R = np.linalg.qr(A, mode = "reduced")

        diag = np.diag(R)
        phase = np.ones_like(diag, dtype = complex)
        nz = np.abs(diag) > 1e-14
        phase[nz] = diag[nz] / np.abs(diag[nz])
        Q = Q * phase.conj()[None, :]

        kraus = [Q[2 * mu : 2 * (mu + 1), :] for mu in range(n_kraus)]
        return kraus


    @staticmethod
    def kraus_tp_error(kraus: list[np.ndarray]):
        accum = np.zeros((2, 2), dtype=complex)
        for K in kraus:
            accum += K.conj().T @ K
        return accum - np.eye(2, dtype=complex)


    @staticmethod
    def initial_identity_params(n_kraus: int = 4, noise: float = 1e-3, seed: int | None = 1):
        """Initial parameters near the identity channel."""
        rng = np.random.default_rng(seed)
        V = np.zeros((2 * n_kraus, 2), dtype=complex)
        V[0:2, :] = np.eye(2, dtype=complex)  # K0 = I, others = 0
        V = V + noise * (rng.normal(size=V.shape) + 1j * rng.normal(size=V.shape))
        return Quantum_EAPT_CPTP._complex_matrix_to_params(V)


    # -----------------------------------------------------------------------------
    # Channel action, Choi matrix, and chi matrix
    # -----------------------------------------------------------------------------


    @staticmethod
    def apply_channel_second_qubit(rho_2q: np.ndarray, kraus: list[np.ndarray]):
        out = np.zeros_like(rho_2q, dtype = complex)
        I = np.eye(2, dtype=complex)
        for K in kraus:
            op = np.kron(I, K)
            out += op @ rho_2q @ op.conj().T
        return 0.5 * (out + out.conj().T)


    @staticmethod
    def kraus_to_choi(kraus: list[np.ndarray]):
        """Return the Choi matrix J_E = sum_mu |K_mu>><<K_mu|.

        Vectorization convention: column stacking, vec(K) = K.reshape(-1, order='F').
        With this convention, trace_out(J_E) = I for a trace-preserving channel.
        """
        J = np.zeros((4, 4), dtype=complex)
        for K in kraus:
            v = K.reshape(-1, order="F")[:, None]
            J += v @ v.conj().T
        return 0.5 * (J + J.conj().T)

    @staticmethod
    def partial_trace_output_from_choi(J: np.ndarray):
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


    @staticmethod
    def choi_trace_preserving_error(J: np.ndarray):
        """Compute Tr_output(J) - I."""
        return Quantum_EAPT_CPTP.partial_trace_output_from_choi(J) - np.eye(2, dtype=complex)


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
    def kraus_to_chi(kraus: list[np.ndarray]):
        """Return process matrix chi in unnormalized Pauli basis {I, X, Y, Z}.

        E(rho) = sum_{m,n} chi[m,n] P_m rho P_n^dagger.

        Since Tr(P_m^dagger P_n)=2 delta_mn, coefficients are
            a_{mu,m} = Tr(P_m^dagger K_mu) / 2.
        """
        P = Quantum_EAPT_CPTP.pauli_stokes_basis()[0]
        chi = np.zeros((4, 4), dtype=complex)
        for K in kraus:
            a = np.array([np.trace(Pm.conj().T @ K) / 2.0 for Pm in P], dtype=complex)
            chi += np.outer(a, a.conj())
        return 0.5 * (chi + chi.conj().T)


    @staticmethod
    def chi_tp_matrix(chi: np.ndarray):
        """Return sum_mn chi_mn P_n^dagger P_m, which equals I for TP."""
        P = Quantum_EAPT_CPTP.pauli_stokes_basis()[0]
        accum = np.zeros((2, 2), dtype=complex)
        for m in range(4):
            for n in range(4):
                accum += chi[m, n] * (P[n].conj().T @ P[m])
        return accum


    # -----------------------------------------------------------------------------
    # MLE prediction and residual
    # -----------------------------------------------------------------------------


    @staticmethod
    def predict_coincidences(
        kraus: list[np.ndarray],
        intensity: float,
        accidentals: np.ndarray | float = 0.0,
        states = STATE_ORDER,
        bell_state_name: str = "Psi+",
    ):
        """Predict a 6x6 coincidence matrix for Psi+ input and a CPTP channel."""
        rho_in = Quantum_EAPT_CPTP.bell_state(name = bell_state_name)
        rho_out = Quantum_EAPT_CPTP.apply_channel_second_qubit(rho_in, kraus)
        projs = Quantum_EAPT_CPTP.projector_list(states)

        pred = np.zeros((len(states), len(states)), dtype=float)
        for i, Pref in enumerate(projs):
            for j, Pch in enumerate(projs):
                M = np.kron(Pref, Pch)
                p = np.real(np.trace(M @ rho_out))
                pred[i, j] = max(p, 0.0) * intensity

        pred = pred + np.asarray(accidentals, dtype=float)
        return np.maximum(pred, 0.01)


    @staticmethod
    def _unpack_fit_params(x: np.ndarray, n_kraus: int) -> tuple[list[np.ndarray], float]:
        n_channel_params = 2 * (2 * n_kraus) * 2
        kraus_params = x[:n_channel_params]
        log_intensity = x[n_channel_params]
        kraus = Quantum_EAPT_CPTP.params_to_kraus(kraus_params, n_kraus=n_kraus)
        intensity = float(np.exp(log_intensity))
        return kraus, intensity

    @staticmethod
    def _mle_residual(
        x: np.ndarray,
        counts: np.ndarray,
        accidentals: np.ndarray,
        n_kraus: int,
        weights: np.ndarray | None,
        bell_state_name: str = "Psi+",
    ):
        kraus, intensity = Quantum_EAPT_CPTP._unpack_fit_params(x, n_kraus=n_kraus)
        pred = Quantum_EAPT_CPTP.predict_coincidences(kraus, intensity, accidentals = accidentals, bell_state_name = bell_state_name)
        residual = (pred - counts) / np.sqrt(pred)
        if weights is not None:
            residual = residual * weights
        return residual.ravel()


    @staticmethod
    def qpt_mle_chi_from_psi_plus_coincidences(
        counts: np.ndarray,
        bell_state_name: str = "Psi+",
        accidentals: np.ndarray | float = 0.0,
        n_kraus: int = 4,
        seed: int | None = 1,
        max_nfev: int = 20000,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        verbose: int = 0,
        weights: np.ndarray | None = None,
    ):
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

        init_kraus_params = Quantum_EAPT_CPTP.initial_identity_params(n_kraus=n_kraus, seed=seed)
        x0 = np.concatenate([init_kraus_params, [np.log(init_intensity)]])

        result = least_squares(
            Quantum_EAPT_CPTP._mle_residual,
            x0,
            args = (counts, accidentals, n_kraus, weights, bell_state_name),
            method = "trf",
            max_nfev = max_nfev,
            ftol = ftol,
            xtol = xtol,
            gtol = gtol,
            verbose = verbose,
        )

        kraus, intensity = Quantum_EAPT_CPTP._unpack_fit_params(result.x, n_kraus=n_kraus)
        predicted = Quantum_EAPT_CPTP.predict_coincidences(kraus, intensity, accidentals=accidentals, bell_state_name = bell_state_name)
        rho_out = Quantum_EAPT_CPTP.apply_channel_second_qubit(Quantum_EAPT_CPTP.bell_state(name = bell_state_name), kraus)
        J = Quantum_EAPT_CPTP.kraus_to_choi(kraus)
        chi = Quantum_EAPT_CPTP.kraus_to_chi(kraus)

        choi_eigs = np.linalg.eigvalsh(J)
        chi_eigs = np.linalg.eigvalsh(chi)
        tp_from_kraus = Quantum_EAPT_CPTP.kraus_tp_error(kraus)
        tp_from_choi = Quantum_EAPT_CPTP.choi_trace_preserving_error(J)
        chi_tp = Quantum_EAPT_CPTP.chi_tp_matrix(chi) - np.eye(2)

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
            return Quantum_EAPT_CPTP.STATE_ORDER, Quantum_EAPT_CPTP.STATE_ORDER
        if shape == (4, 4):
            if name in {"chi", "chi_tp_error_diagnostic"}:
                return Quantum_EAPT_CPTP.PAULI_ORDER, Quantum_EAPT_CPTP.PAULI_ORDER
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
    def _save_complex_matrix_tables(folder: str, prefix: str, name: str, M: np.ndarray):
        """Save a complex matrix as labeled real/imag/abs/phase CSV tables."""
        arr = np.asarray(M)
        row_labels, col_labels = Quantum_EAPT_CPTP._basis_labels_for_matrix(name, arr.shape)
        safe_name = name.replace("/", "_").replace(" ", "_")
    
        if np.iscomplexobj(arr):
            Quantum_EAPT_CPTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_real.csv"), arr.real, row_labels, col_labels)
            Quantum_EAPT_CPTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_imag.csv"), arr.imag, row_labels, col_labels)
            Quantum_EAPT_CPTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_abs.csv"), np.abs(arr), row_labels, col_labels)
            Quantum_EAPT_CPTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}_phase_rad.csv"), np.angle(arr), row_labels, col_labels)
        else:
            Quantum_EAPT_CPTP._save_real_table(os.path.join(folder, f"{prefix}_{safe_name}.csv"), arr, row_labels, col_labels)
    
    
    @staticmethod
    def save_fit_return_values_as_csv_tables(fit: dict, folder: str, prefix: str = "fit"):
        """Save qpt_mle_chi_from_psi_plus_coincidences_cp_only return values as CSV tables.
    
        The function writes labeled CSV files into `folder`:
            - chi, J, rho_out: 4x4 complex tables with HH/HV/VH/VV or Pauli labels
            - predicted: 6x6 real table with H,V,D,A,R,L labels
            - kraus: one 2x2 complex table per Kraus operator
            - cptp_diagnostics: scalar summary plus matrix-valued diagnostics
            - result: least_squares scalar summary and optimized parameter vector
        """
        os.makedirs(folder, exist_ok=True)
    
        # scalar_rows = []
    
        for key, value in fit.items():
            if key == "cptp_diagnostics":
                cp_diagnostics_folder = folder + "/cptp_diagnostics"
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
                        Quantum_EAPT_CPTP._save_complex_matrix_tables(cp_diagnostics_folder, prefix, f"cptp_diagnostics_{dkey}", darr)
                pd.DataFrame(diag_scalar_rows).to_csv(
                    os.path.join(cp_diagnostics_folder, f"{prefix}_cptp_diagnostics_scalars.csv"), index=False
                )
                continue
    
            if key == "kraus":
                for k, K in enumerate(value):
                    kraus_folder = folder + "/Kraus_operators"
                    os.makedirs(kraus_folder, exist_ok=True)
                    Quantum_EAPT_CPTP._save_complex_matrix_tables(kraus_folder, prefix, f"kraus_{k:02d}", np.asarray(K))
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
    
            # arr = np.asarray(value)
            # result_folder = folder + "/result"
            # os.makedirs(result_folder, exist_ok=True)
            # if arr.ndim == 0:
            #     scalar_val = arr.item()
            #     scalar_rows.append({
            #         "quantity": key,
            #         "real": float(np.real(scalar_val)),
            #         "imag": float(np.imag(scalar_val)),
            #     })
            # elif arr.ndim in (1, 2):
            #     Quantum_EAPT_CPTP._save_complex_matrix_tables(result_folder, prefix, key, arr)
            # else:
            #     pd.DataFrame({"flat_index": np.arange(arr.size), "value": arr.reshape(-1)}).to_csv(
            #         os.path.join(result_folder, f"{prefix}_{key}_flattened.csv"), index=False
            #     )
    
        # if scalar_rows:
        #     pd.DataFrame(scalar_rows).to_csv(os.path.join(folder, f"{prefix}_scalars.csv"), index=False)
    
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
