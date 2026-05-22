import numpy as np
import pandas as pd

def pauli_basis_stokes_order(y_sign=+1):
    """
    Pauli basis in Stokes order:
        sigma0 = I
        sigma1 = Z  -> H/V
        sigma2 = X  -> D/A
        sigma3 = Y  -> R/L

    y_sign:
        +1 uses Y = [[0,-i],[i,0]]
        -1 uses -Y, useful if your R/L convention is opposite.
    """

    I = np.array([[1, 0], [0, 1]], dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    return [I, Z, X, y_sign * Y]


def chi_to_mueller(
    chi,
    process_basis="stokes_pauli",
    normalized_process_basis=False,
    y_sign=+1,
    force_real=True,
):
    """
    Convert a QPT chi matrix to a Mueller matrix.

    Assumes:
        E(rho) = sum_{m,n} chi[m,n] E_m rho E_n^dagger

    Parameters
    ----------
    chi : array-like, shape (4,4)
        Process chi matrix.

    process_basis : str
        Currently supports:
            "stokes_pauli": E_m = [I, Z, X, Y]
        This order matches the Stokes vector [S0,S1,S2,S3].

    normalized_process_basis : bool
        False:
            E_m are unnormalized Pauli matrices.
        True:
            E_m = sigma_m / sqrt(2).

        Many QPT packages use unnormalized Pauli basis,
        but some use normalized basis. This changes scale.

    y_sign : +1 or -1
        Controls R/L convention.
        Use -1 if your circular polarization convention is opposite.

    force_real : bool
        If True, return real part after checking numerical imaginary residue.

    Returns
    -------
    M : np.ndarray, shape (4,4)
        Mueller matrix satisfying S_out = M S_in.
    """

    chi = np.asarray(chi, dtype=complex)

    if chi.shape != (4, 4):
        raise ValueError("chi must be a 4x4 matrix.")

    if process_basis != "stokes_pauli":
        raise NotImplementedError("Only process_basis='stokes_pauli' is implemented.")

    sigma = pauli_basis_stokes_order(y_sign=y_sign)

    if normalized_process_basis:
        E = [s / np.sqrt(2.0) for s in sigma]
    else:
        E = sigma

    M = np.zeros((4, 4), dtype=complex)

    for i in range(4):
        for j in range(4):
            val = 0.0 + 0.0j

            for m in range(4):
                for n in range(4):
                    val += chi[m, n] * np.trace(
                        sigma[i] @ E[m] @ sigma[j] @ E[n].conj().T
                    )

            M[i, j] = 0.5 * val

    if force_real:
        imag_norm = np.linalg.norm(np.imag(M))
        if imag_norm > 1e-8:
            print(f"Warning: Mueller matrix has non-negligible imaginary part: {imag_norm:.3e}")
        M = np.real(M)

    return M

def chi_complex_recon(real,imag):
    """
    Reconstruct complex chi matrix from separate real and imaginary parts.

    Parameters
    ----------
    real : array-like, shape (4,4)
        Real part of chi matrix.
    imag : array-like, shape (4,4)
        Imaginary part of chi matrix.

    Returns
    -------
    chi : np.ndarray, shape (4,4)
        Complex chi matrix.
    """

    real = np.asarray(real)
    imag = np.asarray(imag)

    if real.shape != (4, 4) or imag.shape != (4, 4):
        raise ValueError("real and imag must be 4x4 matrices.")

    return real + 1j * imag

Folder_name = "Reference_20260507_labeled_QPT_analysis_NTP_class"
Data_file_name = "Reference_20260507"
Path_name = Folder_name + "/result/"
Data_file_name_real = Data_file_name + "_chi_real"
Data_file_name_imag = Data_file_name + "_chi_imag"
# print(Path_name + Data_file_name_real + ".csv")
chi_real = pd.read_csv(Path_name + Data_file_name_real + ".csv", delimiter=",")
# chi_imag = pd.read_csv(Path_name + Data_file_name_imag + ".csv", delimiter=",")