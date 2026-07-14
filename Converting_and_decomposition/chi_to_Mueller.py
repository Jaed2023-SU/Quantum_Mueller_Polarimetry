import numpy as np
import pandas as pd

def load_chi_from_csv(real_csv_path, imag_csv_path):

    chi_real = pd.read_csv(real_csv_path, index_col=0).to_numpy(dtype=float)
    chi_imag = pd.read_csv(imag_csv_path, index_col=0).to_numpy(dtype=float)

    chi = chi_real + 1j * chi_imag

    return chi

def chi2mueller(chi, Pauli_basis, Stokes_basis):
    if chi.shape != (4, 4):
        raise ValueError(f"chi must be 4x4, got {chi.shape}")

    M_complex = np.zeros((4, 4), dtype=complex)

    for mu in range(4):
        for nu in range(4):
            value = 0.0 + 0.0j

            for m in range(4):
                for n in range(4):
                    value += (
                        chi[m, n]
                        * np.trace(
                            Stokes_basis[mu]
                            @ Pauli_basis[m]
                            @ Stokes_basis[nu].conj().T
                            @ Pauli_basis[n].conj().T
                        )
                    )

            M_complex[mu, nu] = 0.5 * value
            
    if np.sum(np.imag(M_complex)) > 10e-10:
        raise ValueError("Mueller matrix should be a real matrix")
    M = np.real(M_complex)
    M /= M[0, 0]

    return M

def save_mueller_csv(M, output_csv_path):
    if M.shape != (4, 4):
        raise ValueError(f"Mueller matrix must be 4x4, got {M.shape}")

    np.savetxt(output_csv_path, M, delimiter=",", fmt="%.10g")