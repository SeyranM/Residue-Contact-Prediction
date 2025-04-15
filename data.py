import os
import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd

class PreprocessedTensorDataset(Dataset):
    def __init__(self, tensor_dir: str):
        self.tensor_files: List[str] = [
            os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir)[:4000]
            if f.endswith(".pt")
        ]
        self.lengths = self._load_lengths()

    def _load_lengths(self):
        lengths_path = os.path.join(os.path.dirname(self.tensor_files[0]), "../sequence_lengths.csv")
        lengths_path = os.path.abspath(lengths_path)
        df = pd.read_csv(lengths_path)
        file_to_len = {row["file"]: row["length"] for _, row in df.iterrows()}
        lengths = []
        for path in self.tensor_files:
            file_id = os.path.basename(path).replace(".pt", "")
            lengths.append(file_to_len.get(file_id, 0))
        return lengths

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        path = self.tensor_files[idx]
        data = torch.load(path)
        return {
            "token_embeddings": data["token_embeddings"],
            "idx1": data["idx1"],
            "idx2": data["idx2"],
            "labels": data["labels"],
            "residue_pairs": data["residue_pairs"],
            "file_name": os.path.basename(path).replace(".pt", ""),
            "most_similar_contact_map": data.get("most_similar_contact_map")
        }

    def get_seq_lengths(self):
        return self.lengths
