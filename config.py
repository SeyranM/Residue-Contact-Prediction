import torch
from pathlib import Path


class CFG:
    esm_model_name = "facebook/esm2_t12_35M_UR50D"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parent_dir = Path(__file__).parent
    logs_dir = parent_dir.joinpath("logs")
    data_dir = parent_dir.joinpath("data")
    models_dir = parent_dir.joinpath("models")
    reporting_dir = parent_dir.joinpath("reporting")
    visualization_dir = parent_dir.joinpath("visualizations")

    best_model_path = models_dir.joinpath("contact_model_f1_69.61_epoch_19.pt")

    raw_data_dir = data_dir.joinpath("raw_data")
    intermediate_data_dir = data_dir.joinpath("intermediate_data")
    processed_data_dir = data_dir.joinpath("processed_data")

    # 3-letter to 1-letter amino acid code mapping
    mapping = {
        "ALA": "A", "ARG": "R", "ASP": "D", "CYS": "C", "CYX": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H",
        "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "ASN": "N",
        "PHE": "F", "PRO": "P", "SEC": "U", "SER": "S", "THR": "T",
        "TRP": "W", "TYR": "Y", "VAL": "V"
    }

    lr = 1e-4
    epochs = 30
    embedding_dim = 480
    val_batch_size = 4
    train_batch_size = 4
    val_bucket_size = 2000
    train_bucket_size = 2000
    max_sequence_length = 750
    atoms_distance_threshold = 8
