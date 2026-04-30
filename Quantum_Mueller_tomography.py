import csv
import numpy as np


LABELS = ["H", "V", "D", "A", "R", "L"]


# ============================================================
# 1. CSV loader
# ============================================================

def load_36_counts_csv(path, labels=LABELS):
    """
    Load 36 coincidence counts N_WU from CSV.

    Supported CSV formats:

    Format A: 6 x 6 numeric matrix
        rows: photon-1 projection W = H,V,D,A,R,L
        columns: photon-2 projection U = H,V,D,A,R,L

        Example:
            123, 120, ...
            ...

    Format B: 7 x 7 table with row/column labels
            ,H,V,D,A,R,L
        H,  NHH,NHV,NHD,NHA,NHR,NHL
        V,  NVH,NVV,...

    Format C: 36 rows, 3 columns
        W,U,count
        H,H,123
        H,V,120
        ...

    Format D: 36 rows, 2 columns
        label,count
        HH,123
        HV,120
        ...

    Returns
    -------
    N : dict
        N[(W,U)] = coincidence count
    C : np.ndarray, shape (6,6)
        C[i,j] = N[(labels[i], labels[j])]
    """

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    # Remove empty rows and strip spaces
    rows = [[cell.strip() for cell in row] for row in rows if len(row) > 0]
    rows = [row for row in rows if any(cell != "" for cell in row)]

    # Try to remove header-like row for 36-row formats
    def is_float(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    # ---------- Format B: 7x7 labeled matrix ----------
    if len(rows) >= 7 and len(rows[0]) >= 7:
        header = rows[0]
        if all(x in labels or x == "" for x in header[:7]):
            col_labels = header[1:7]
            if set(col_labels) == set(labels):
                C = np.zeros((6, 6), dtype=float)
                for i in range(6):
                    row_label = rows[i + 1][0]
                    if row_label not in labels:
                        raise ValueError("Invalid row label in labeled 7x7 CSV.")
                    for j in range(6):
                        C[labels.index(row_label), labels.index(col_labels[j])] = float(rows[i + 1][j + 1])
                N = {(labels[i], labels[j]): C[i, j] for i in range(6) for j in range(6)}
                return N, C

    # ---------- Format A: 6x6 numeric matrix ----------
    if len(rows) == 6 and all(len(row) >= 6 for row in rows):
        if all(is_float(rows[i][j]) for i in range(6) for j in range(6)):
            C = np.array([[float(rows[i][j]) for j in range(6)] for i in range(6)], dtype=float)
            N = {(labels[i], labels[j]): C[i, j] for i in range(6) for j in range(6)}
            return N, C

    # ---------- Format C or D: row list ----------
    # Remove header if first row is non-data
    data_rows = rows[:]
    if len(data_rows) > 0:
        first = data_rows[0]
        if len(first) >= 2 and not is_float(first[-1]):
            data_rows = data_rows[1:]

    N = {}

    for row in data_rows:
        # Format C: W,U,count
        if len(row) >= 3 and row[0] in labels and row[1] in labels and is_float(row[2]):
            W, U, val = row[0], row[1], float(row[2])
            N[(W, U)] = val

        # Format D: label,count; e.g. HH,123
        elif len(row) >= 2 and len(row[0]) == 2 and row[0][0] in labels and row[0][1] in labels and is_float(row[1]):
            W, U, val = row[0][0], row[0][1], float(row[1])
            N[(W, U)] = val

    missing = [(W, U) for W in labels for U in labels if (W, U) not in N]
    if missing:
        raise ValueError(f"Missing coincidence counts for: {missing}")

    C = np.array([[N[(W, U)] for U in labels] for W in labels], dtype=float)
    return N, C


# ============================================================
# 2. Mueller matrix from 36 coincidence counts
#    Table 3 of the paper, for |Phi+>
# ============================================================

def mueller_from_phi_plus_counts(N, normalize=True):
    """
    Compute Mueller matrix from 36 coincidence counts N_WU
    for the |Phi+> Bell-state nonclassical Mueller polarimetry case.

    N[(W,U)]:
        W = photon 1 projection
        U = photon 2 projection

    This implements Table 3 of the paper.

    Returns
    -------
    M : np.ndarray, shape (4,4)
        Mueller matrix. If normalize=True, M[0,0] = 1.
    """

    def n(W, U):
        return float(N[(W, U)])

    m = np.zeros((4, 4), dtype=float)

    # Row 1
    m[0, 0] = n("H", "H") + n("H", "V") + n("V", "H") + n("V", "V")
    m[0, 1] = n("H", "H") + n("H", "V") - n("V", "H") - n("V", "V")
    m[0, 2] = n("D", "H") + n("D", "V") - n("A", "H") - n("A", "V")
    m[0, 3] = n("L", "H") + n("L", "V") - n("R", "H") - n("R", "V")

    # Row 2
    m[1, 0] = n("H", "H") + n("V", "H") - n("H", "V") - n("V", "V")
    m[1, 1] = n("H", "H") + n("V", "V") - n("V", "H") - n("H", "V")
    m[1, 2] = n("D", "H") + n("A", "V") - n("A", "H") - n("D", "V")
    m[1, 3] = n("L", "H") + n("R", "V") - n("R", "H") - n("L", "V")

    # Row 3
    m[2, 0] = n("H", "D") + n("V", "D") - n("V", "A") - n("H", "A")
    m[2, 1] = n("H", "D") + n("V", "A") - n("V", "D") - n("H", "A")
    m[2, 2] = n("A", "A") + n("D", "D") - n("D", "A") - n("A", "D")
    m[2, 3] = n("L", "D") + n("R", "A") - n("L", "A") - n("R", "D")

    # Row 4
    m[3, 0] = n("H", "R") + n("V", "R") - n("H", "L") - n("V", "L")
    m[3, 1] = n("H", "R") + n("V", "L") - n("V", "R") - n("H", "L")
    m[3, 2] = n("A", "L") + n("D", "R") - n("A", "R") - n("D", "L")
    m[3, 3] = n("R", "L") + n("L", "R") - n("L", "L") - n("R", "R")

    if normalize:
        if abs(m[0, 0]) < 1e-15:
            raise ValueError("m11 is zero or too small; cannot normalize Mueller matrix.")
        m = m / m[0, 0]

    return m


# ============================================================
# 3. Matrix utilities
# ============================================================

def symmetrize(A):
    return 0.5 * (A + A.T)


def sqrtm_psd(A, eps=1e-12):
    """
    Symmetric positive-semidefinite matrix square root.
    """
    A = symmetrize(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def inv_sqrtm_psd(A, eps=1e-12):
    """
    Inverse square root for symmetric PSD matrix.
    Small eigenvalues are regularized.
    """
    A = symmetrize(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def nearest_rotation_matrix(R):
    """
    Project a 3x3 matrix onto the nearest proper rotation matrix SO(3).
    """
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


# ============================================================
# 4. Lu-Chipman-style polar decomposition
# ============================================================

def lu_chipman_decomposition(M, eps=1e-12):
    """
    Decompose Mueller matrix into:
        M ≈ M_delta @ M_R @ M_D

    where
        M_D     : diattenuation matrix
        M_R     : retardance matrix
        M_delta : depolarization / polarizance-like matrix

    Note:
        The paper writes the conceptual decomposition as M = D R P.
        In many Mueller-polarimetry references, the forward Lu-Chipman
        decomposition is written as M = M_delta M_R M_D.
        This function returns the physically standard components with names.

    Returns
    -------
    result : dict
        {
            "M": normalized Mueller matrix,
            "diattenuation_matrix": M_D,
            "retardance_matrix": M_R,
            "polarizance_matrix": M_delta,
            "diattenuation_vector": d,
            "polarizance_vector": p,
            "diattenuation_magnitude": Dmag,
            "depolarization_index_delta": Delta
        }
    """

    M = np.array(M, dtype=float).copy()

    if abs(M[0, 0]) < eps:
        raise ValueError("M[0,0] is zero or too small.")
    M = M / M[0, 0]

    # Diattenuation vector from first row
    d = M[0, 1:4].copy()
    Dmag = np.linalg.norm(d)

    # Avoid numerical problems for nearly ideal polarizers
    Dclip = min(Dmag, 1.0 - 1e-10)

    if Dmag > eps:
        dhat = d / Dmag
        sqrt_term = np.sqrt(max(0.0, 1.0 - Dclip**2))
        mD = sqrt_term * np.eye(3) + (1.0 - sqrt_term) * np.outer(dhat, dhat)
    else:
        mD = np.eye(3)

    M_D = np.eye(4)
    M_D[0, 1:4] = d
    M_D[1:4, 0] = d
    M_D[1:4, 1:4] = mD

    # Remove diattenuator from the right
    # Use pseudo-inverse to handle highly diattenuating cases.
    M_prime = M @ np.linalg.pinv(M_D)

    # M_prime ≈ M_delta @ M_R
    p_delta = M_prime[1:4, 0].copy()
    A = M_prime[1:4, 1:4].copy()

    # Left polar decomposition:
    # A = m_delta @ R
    m_delta = sqrtm_psd(A @ A.T, eps=eps)
    R3 = np.linalg.pinv(m_delta) @ A
    R3 = nearest_rotation_matrix(R3)

    M_R = np.eye(4)
    M_R[1:4, 1:4] = R3

    M_delta = np.eye(4)
    M_delta[1:4, 0] = p_delta
    M_delta[1:4, 1:4] = m_delta

    # Polarizance vector of original M
    p = M[1:4, 0].copy()

    # Paper-style depolarization indicator:
    # Delta = 1 - |Tr(P)-1|/3, where P here corresponds to M_delta.
    Delta = 1.0 - abs(np.trace(M_delta) - 1.0) / 3.0

    return {
        "M": M,
        "diattenuation_matrix": M_D,
        "retardance_matrix": M_R,
        "polarizance_matrix": M_delta,
        "diattenuation_vector": d,
        "polarizance_vector": p,
        "diattenuation_magnitude": Dmag,
        "depolarization_index_delta": Delta,
    }


# ============================================================
# 5. Retardance, birefringence, and optical-axis extraction
# ============================================================

def retardance_axis_from_retarder(M_R, wavelength_m=None, thickness_m=None, eps=1e-12):
    """
    Extract retardance and optical axis information from the retardance matrix M_R.

    Parameters
    ----------
    M_R : np.ndarray, shape (4,4)
        Retardance Mueller matrix.
    wavelength_m : float or None
        Optical wavelength in meters.
    thickness_m : float or None
        Sample thickness in meters.

    Returns
    -------
    info : dict
        retardance_rad
        retardance_deg
        poincare_axis
        linear_retarder_axis_deg
        birefringence_abs
        notes

    Notes
    -----
    The 3x3 block of a pure retarder is a rotation on the Poincare sphere.
    Its rotation angle is the retardance magnitude.

    For a purely linear retarder:
        Poincare axis ≈ (cos 2theta, sin 2theta, 0)
    so
        theta = 0.5 atan2(axis_S2, axis_S1)

    Absolute fast/slow labeling requires a sign convention and calibration.
    Mueller intensity data alone usually gives the magnitude robustly,
    while fast-vs-slow assignment is convention-dependent.
    """

    R = np.array(M_R[1:4, 1:4], dtype=float)
    R = nearest_rotation_matrix(R)

    cos_delta = (np.trace(R) - 1.0) / 2.0
    cos_delta = np.clip(cos_delta, -1.0, 1.0)
    delta = np.arccos(cos_delta)

    if abs(np.sin(delta)) > eps:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(delta))
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([np.nan, np.nan, np.nan])

    # Linear retarder axis angle on lab H/V basis
    # valid when axis[2] ≈ 0
    if np.all(np.isfinite(axis)):
        linear_axis_rad = 0.5 * np.arctan2(axis[1], axis[0])
        linear_axis_deg = np.degrees(linear_axis_rad)

        # Map to [-90, 90)
        while linear_axis_deg >= 90.0:
            linear_axis_deg -= 180.0
        while linear_axis_deg < -90.0:
            linear_axis_deg += 180.0
    else:
        linear_axis_deg = np.nan

    birefringence_abs = None
    if wavelength_m is not None and thickness_m is not None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive.")
        birefringence_abs = delta * wavelength_m / (2.0 * np.pi * thickness_m)

    notes = (
        "retardance_rad is the phase retardance magnitude. "
        "birefringence_abs = |Delta n| if wavelength_m and thickness_m are supplied. "
        "linear_retarder_axis_deg is meaningful when the Poincare axis S3 component is near zero. "
        "Fast/slow-axis assignment requires an external sign convention or calibration."
    )

    return {
        "retardance_rad": delta,
        "retardance_deg": np.degrees(delta),
        "poincare_axis": axis,
        "linear_retarder_axis_deg": linear_axis_deg,
        "birefringence_abs": birefringence_abs,
        "notes": notes,
    }


# ============================================================
# 6. Full pipeline
# ============================================================

def analyze_nonclassical_mueller_csv(
    csv_path,
    wavelength_nm=None,
    thickness_um=None,
    labels=LABELS,
):
    """
    Full analysis pipeline.

    Parameters
    ----------
    csv_path : str
        CSV file containing 36 coincidence counts.
    wavelength_nm : float or None
        Wavelength in nm. Needed only for birefringence |Delta n|.
    thickness_um : float or None
        Sample thickness in micrometer. Needed only for birefringence |Delta n|.

    Returns
    -------
    result : dict
        Includes:
            total_mueller_matrix
            diattenuation_matrix
            retardance_matrix
            polarizance_matrix
            retardance_info
            raw_counts_matrix_6x6
    """

    N, C = load_36_counts_csv("Mueller_sample_20260428.csv", labels=labels)

    M = mueller_from_phi_plus_counts(N, normalize=True)

    decomp = lu_chipman_decomposition(M)

    wavelength_m = None if wavelength_nm is None else wavelength_nm * 1e-9
    thickness_m = None if thickness_um is None else thickness_um * 1e-6

    ret_info = retardance_axis_from_retarder(
        decomp["retardance_matrix"],
        wavelength_m=wavelength_m,
        thickness_m=thickness_m,
    )

    return {
        "raw_counts_matrix_6x6": C,
        "total_mueller_matrix": decomp["M"],
        "diattenuation_matrix": decomp["diattenuation_matrix"],
        "retardance_matrix": decomp["retardance_matrix"],
        "polarizance_matrix": decomp["polarizance_matrix"],
        "diattenuation_vector": decomp["diattenuation_vector"],
        "polarizance_vector": decomp["polarizance_vector"],
        "diattenuation_magnitude": decomp["diattenuation_magnitude"],
        "depolarization_index_delta": decomp["depolarization_index_delta"],
        "retardance_info": ret_info,
    }


# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":
    csv_path = "coincidence_counts.csv"

    # Example:
    # wavelength_nm = 810 for the experiment in the paper.
    # thickness_um should be supplied only if the sample thickness is known.
    result = analyze_nonclassical_mueller_csv(
        csv_path,
        wavelength_nm=810.0,
        thickness_um=None,
    )

    print("\nTotal Mueller matrix M:")
    print(np.array2string(result["total_mueller_matrix"], precision=5, suppress_small=True))

    print("\nDiattenuation matrix M_D:")
    print(np.array2string(result["diattenuation_matrix"], precision=5, suppress_small=True))

    print("\nRetardance matrix M_R:")
    print(np.array2string(result["retardance_matrix"], precision=5, suppress_small=True))

    print("\nPolarizance / depolarization matrix M_delta:")
    print(np.array2string(result["polarizance_matrix"], precision=5, suppress_small=True))

    print("\nDiattenuation magnitude:")
    print(result["diattenuation_magnitude"])

    print("\nDepolarization index Delta:")
    print(result["depolarization_index_delta"])

    print("\nRetardance / axis information:")
    for k, v in result["retardance_info"].items():
        print(f"{k}: {v}")