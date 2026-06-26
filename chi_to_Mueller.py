import numpy as np
import pandas as pd
from Quantum_EAPT_NTP import Quantum_EAPT_NTP

QPT = Quantum_EAPT_NTP


def load_chi_from_csv(real_csv_path, imag_csv_path):

    chi_real = pd.read_csv(real_csv_path, index_col=0).to_numpy(dtype=float)
    chi_imag = pd.read_csv(imag_csv_path, index_col=0).to_numpy(dtype=float)

    chi = chi_real + 1j * chi_imag

    return chi


def chi_to_mueller_hv_stokes(chi):
    if chi.shape != (4, 4):
        raise ValueError(f"chi must be 4x4, got {chi.shape}")

    # Pauli basis used in chi matrix: (I, X, Y, Z)
    I = np.array([[1, 0],
                  [0, 1]], dtype=complex)

    X = np.array([[0, 1],
                  [1, 0]], dtype=complex)

    Y = np.array([[0, -1j],
                  [1j, 0]], dtype=complex)

    Z = np.array([[1, 0],
                  [0, -1]], dtype=complex)

    pauli_basis = [I, X, Y, Z]

    # Stokes basis in H/V convention: (S0, S1, S2, S3) = (I, Z, X, Y)
    stokes_basis = [I, Z, X, Y]

    M_complex = np.zeros((4, 4), dtype=complex)

    for mu in range(4):
        for nu in range(4):
            value = 0.0 + 0.0j

            for m in range(4):
                for n in range(4):
                    value += (
                        chi[m, n]
                        * np.trace(
                            stokes_basis[mu]
                            @ pauli_basis[m]
                            @ stokes_basis[nu]
                            @ pauli_basis[n].conj().T
                        )
                    )

            M_complex[mu, nu] = 0.5 * value

    # Mueller matrix should be real for a physical chi matrix.
    # Small imaginary parts can appear from numerical error.
    M = np.real_if_close(M_complex, tol=1000).real

    return M


def save_mueller_to_csv(M, output_csv_path):
    """
    Save 4x4 real Mueller matrix to CSV.

    Parameters
    ----------
    M : np.ndarray
        Real 4x4 Mueller matrix.
    output_csv_path : str
        Output CSV file path.
    """
    if M.shape != (4, 4):
        raise ValueError(f"Mueller matrix must be 4x4, got {M.shape}")

    np.savetxt(output_csv_path, M, delimiter=",", fmt="%.10g")


def convert_chi_csv_to_mueller_csv(real_csv_path, imag_csv_path, output_csv_path):
    """
    Full pipeline:
        chi_real.csv + chi_imag.csv
        -> complex chi ndarray
        -> real Mueller ndarray
        -> Mueller CSV
    """
    chi = load_chi_from_csv(real_csv_path, imag_csv_path)
    M = chi_to_mueller_hv_stokes(chi)
    save_mueller_to_csv(M, output_csv_path)

    return chi, M


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    real_csv = "Reference_20260507_chi_real.csv"
    imag_csv = "Reference_20260507_chi_imag.csv"
    output_csv = "Mueller_Reference_20260507.csv"

    chi, M = convert_chi_csv_to_mueller_csv(
        real_csv,
        imag_csv,
        output_csv
    )

    print("Loaded chi matrix:")
    print(chi)

    print("\nConverted Mueller matrix:")
    print(M)

    print(f"\nSaved Mueller matrix to: {output_csv}")
    
    
    QPT.plot_chi(

    M,

    save_name = "Mueller_Reference_20260507"

    )