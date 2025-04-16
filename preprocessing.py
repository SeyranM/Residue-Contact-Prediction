import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from loguru import logger
from sklearn.model_selection import train_test_split
from config import CFG
from sklearn.metrics.pairwise import cosine_similarity


class ContactMapPreprocessor:
    def __init__(self, n_processes: int = cpu_count() // 2):
        self.n_processes = n_processes

    def extract_sequence_and_coords(self, pdb_path: str) -> Tuple[str, np.ndarray]:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        seq = []
        coords = []
        for res in structure.get_residues():
            if res.get_id()[0] != ' ':
                continue
            if 'CA' not in res:
                continue
            resname = res.get_resname()
            if resname not in CFG.mapping:
                continue
            seq.append(CFG.mapping[resname])
            coords.append(res['CA'].get_coord())
        return ''.join(seq), np.array(coords)

    def compute_contact_map(self, coords: np.ndarray) -> np.ndarray:
        if coords.shape[0] < 2:
            return np.zeros((coords.shape[0], coords.shape[0]))
        dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
        return (dist_matrix < CFG.atoms_distance_threshold).astype(int)

    def _process_file(self, args):
        pdb_file, pdb_dir, output_dir = args
        pdb_path = os.path.join(pdb_dir, pdb_file)
        try:
            sequence, coords = self.extract_sequence_and_coords(pdb_path)
            if len(sequence) < 2 or len(sequence) > CFG.max_sequence_length:
                logger.warning(f"Skipping {pdb_file} due to invalid sequence length: {len(sequence)}")
                return
            contact_map = self.compute_contact_map(coords)
            L = len(sequence)
            idx_i, idx_j = np.triu_indices(L, k=1)
            pairs = [(i, j) for i, j in zip(idx_i.tolist(), idx_j.tolist())]
            labels = contact_map[idx_i, idx_j].astype(int).tolist()
            residue_pairs = [f"{sequence[i]}-{sequence[j]}" for i, j in pairs]

            json_output = {
                "sequence": sequence,
                "residue_pairs": residue_pairs,
                "pairs_idxs": pairs,
                "labels": labels
            }
            with open(os.path.join(output_dir, pdb_file.replace(".pdb", ".json")), 'w') as f:
                json.dump(json_output, f)
        except Exception as e:
            logger.error(f"Failed to process {pdb_file}: {e}")

    def run(self):
        pdb_dir = CFG.raw_data_dir.joinpath("train")
        output_dir = CFG.intermediate_data_dir.joinpath("train")
        if output_dir.exists():
            logger.info("Removing existing folder to create new one.")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
        args_list = [(pdb_file, pdb_dir, output_dir) for pdb_file in pdb_files]

        with Pool(processes=self.n_processes) as pool:
            list(tqdm(pool.imap(self._process_file, args_list), total=len(args_list)))


class EmbeddingPreprocessor:
    def __init__(self, batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.esm_model_name)
        self.model = AutoModel.from_pretrained(CFG.esm_model_name).to(CFG.device)
        self.model.eval()
        self.batch_size = batch_size
        self.train_embeddings_df = None

    def load_embeddings_df(self, df_path):
        if os.path.exists(df_path):
            self.train_embeddings_df = pd.read_pickle(df_path)
            self.train_embeddings_df["embedding"] = self.train_embeddings_df["embedding"].apply(lambda x: np.array(x))

    def find_most_similar_file(self, query_embedding, current_file, mode):
        if self.train_embeddings_df is None:
            return None

        all_embeddings = np.stack(self.train_embeddings_df["embedding"].values)
        similarities = cosine_similarity([query_embedding], all_embeddings).flatten()
        sorted_indices = np.argsort(-similarities)

        k = 1 if mode == "val" else 2
        target_idx = sorted_indices[k - 1]
        return self.train_embeddings_df.iloc[target_idx]["file"]

    def process_batch(self, batch_paths: List[str], json_dir: str, output_dir: str, mode: str):
        sequences = []
        data_records = []
        lengths_records = []

        for file in batch_paths:
            json_path = os.path.join(json_dir, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            sequences.append(data["sequence"])
            data_records.append((data, file))

        encodings = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_sequence_length
        )

        input_ids = encodings["input_ids"].to(CFG.device)
        attention_mask = encodings["attention_mask"].to(CFG.device)

        with torch.no_grad():
            token_embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        attention_mask_exp = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        masked_outputs = token_embeddings * attention_mask_exp
        sum_embeddings = masked_outputs.sum(dim=1)
        lengths = attention_mask.sum(dim=1).unsqueeze(1)
        pooled_embeddings = sum_embeddings / lengths

        embedding_records = []

        for i, (data, file) in enumerate(data_records):
            L = len(data["sequence"])
            output = {
                "input_ids": input_ids[i, :L].cpu(),
                "attention_mask": attention_mask[i, :L].cpu(),
                "token_embeddings": token_embeddings[i, :L],
                "idx1": torch.tensor([x[0] for x in data["pairs_idxs"]], dtype=torch.long),
                "idx2": torch.tensor([x[1] for x in data["pairs_idxs"]], dtype=torch.long),
                "labels": torch.tensor(data["labels"], dtype=torch.float),
                "residue_pairs": data["residue_pairs"]
            }

            file_id = file.replace(".json", "")
            lengths_records.append({"file": file_id, "length": L})

            torch.save(output, os.path.join(output_dir, file.replace(".json", ".pt")))

            embedding_records.append({
                "file": file_id,
                "embedding": pooled_embeddings[i].cpu().numpy()
            })

        return embedding_records, lengths_records

    def build_contact_map_from_labels(self, data):
        L = data["token_embeddings"].shape[0]
        contact_map = torch.zeros((L, L), dtype=torch.float)
        for (i, j), label in zip(zip(data["idx1"], data["idx2"]), data["labels"]):
            contact_map[i, j] = label
            contact_map[j, i] = label
        return contact_map

    def _add_single_prior(self, args):
        f, mode = args
        file_id = f.replace(".pt", "")
        pt_path = os.path.join(CFG.processed_data_dir.joinpath(mode), f)
        data = torch.load(pt_path)
        if self.train_embeddings_df is not None:
            query_embedding = self.train_embeddings_df[self.train_embeddings_df["file"] == file_id]["embedding"].values[0]
            most_similar_file = self.find_most_similar_file(query_embedding, file_id, mode)
            if most_similar_file:
                similar_path = CFG.processed_data_dir.joinpath("train", f"{most_similar_file}.pt")
                if similar_path.exists():
                    similar_data = torch.load(similar_path)
                    contact_map = self.build_contact_map_from_labels(similar_data)
                    data["most_similar_contact_map"] = contact_map
                    torch.save(data, pt_path)

    def add_similarity_priors(self, mode: str):
        logger.info(f"Adding similarity-based contact priors with multiprocessing for {mode}...")
        processed_dir = CFG.processed_data_dir.joinpath(mode)
        pt_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]
        with Pool(processes=cpu_count() // 2) as pool:
            list(tqdm(pool.imap(self._add_single_prior, [(f, mode) for f in pt_files]),
                      total=len(pt_files),
                      smoothing=0))

    def run(self):
        json_dir = CFG.intermediate_data_dir.joinpath("train")
        processed_dir = CFG.processed_data_dir
        train_dir = processed_dir.joinpath("train")
        val_dir = processed_dir.joinpath("val")

        for d in [train_dir, val_dir]:
            if d.exists():
                logger.info(f"Removing existing folder {d} to create new one.")
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        train_files, val_files = train_test_split(json_files, test_size=0.2, random_state=42)

        self.load_embeddings_df(CFG.processed_data_dir.joinpath("esm2_embeddings_train.pkl"))

        all_embeddings = []
        all_lengths = []

        for split_name, split_files, out_dir in [("train", train_files, train_dir), ("val", val_files, val_dir)]:
            for i in tqdm(range(0, len(split_files), self.batch_size), desc=f"Encoding {split_name}"):
                batch = split_files[i:i+self.batch_size]
                batch_embeddings, batch_lengths = self.process_batch(batch, json_dir, out_dir, mode=split_name)
                all_embeddings.extend(batch_embeddings)
                all_lengths.extend(batch_lengths)

        df = pd.DataFrame(all_embeddings)
        df.to_pickle(processed_dir.joinpath("esm2_embeddings_train.pkl"))

        df_lengths = pd.DataFrame(all_lengths)
        df_lengths.to_csv(processed_dir.joinpath("sequence_lengths.csv"), index=False)

if __name__ == '__main__':
    contact_map_processor = ContactMapPreprocessor(n_processes=8)
    contact_map_processor.run()
    embedding_processor = EmbeddingPreprocessor(batch_size=8)
    embedding_processor.run()
    embedding_processor.add_similarity_priors("train")
    embedding_processor.add_similarity_priors("val")
