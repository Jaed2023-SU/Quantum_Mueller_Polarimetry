from Quantum_EAPT_CPTP import Quantum_EAPT_CPTP
import os

QPT = Quantum_EAPT_CPTP

Data_file_name_raw = "QWPCW14deg_HWPCCW12deg_20260507"
Source_state = "Psi+"
Data_file_name = Data_file_name_raw + "_labeled"
csv_path = Data_file_name + ".csv"
QPT.header_from_raw(Data_file_name_raw + ".csv", csv_path)
Folder_name = Data_file_name + "_QPT_analysis_CPTP_rotate_20260713"
os.makedirs(Folder_name, exist_ok=True)
Coincidence_data = QPT.load_labeled_coincidence_csv(

    csv_path,

    reference_col="Idler",

    channel_col="Signal",

    count_col="count",

    output_matrix_csv=Folder_name + "/" + Data_file_name + "_matrix_labeled.csv",

)

fit = QPT.qpt_mle_chi_from_psi_plus_coincidences(

    Coincidence_data,

    bell_state_name = Source_state,

)

Pauli, Stokes, Pauli_labels, Stokes_labels = QPT.pauli_stokes_basis()

QPT.plot_matrix(fit["chi"], Pauli_labels, title_prefix = "$\chi$ $matrix$",save_name = Folder_name + "/figure_" + Data_file_name)

QPT.save_fit_return_values_as_csv_tables(

    fit,

    folder = Folder_name,

    prefix = Data_file_name_raw,

)