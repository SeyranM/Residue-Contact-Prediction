import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from config import CFG
from model import PairwiseContactModel
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix

def extract_sequence_and_ca_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    seq = []
    coords = []
    for res in structure.get_residues():
        if res.get_id()[0] != ' ' or 'CA' not in res:
            continue
        resname = res.get_resname()
        if resname in CFG.mapping:
            seq.append(CFG.mapping[resname])
            coords.append(res['CA'].get_coord())
    return ''.join(seq), np.array(coords)

def compute_contact_map(coords):
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    contact_map = (dist_matrix < CFG.atoms_distance_threshold).astype(int)
    return contact_map

def evaluate_test_set():
    logger.info("Loading model...")
    model = PairwiseContactModel(embedding_dim=CFG.embedding_dim, best_model_path=CFG.best_model_path).to(CFG.device)
    model.load_state_dict(torch.load(model.best_model_path, map_location=CFG.device))
    model.eval()

    esm_model = AutoModel.from_pretrained(CFG.esm_model_name).to(CFG.device)
    tokenizer = AutoTokenizer.from_pretrained(CFG.esm_model_name)
    esm_model.eval()

    test_dir = CFG.raw_data_dir.joinpath("test")
    pdb_files = [f for f in os.listdir(test_dir) if f.endswith(".pdb")]

    all_preds, all_probs, all_labels = [], [], []
    per_protein_records = []

    for pdb_file in tqdm(pdb_files, desc="Evaluating test PDBs"):
        try:
            seq, coords = extract_sequence_and_ca_coords(test_dir / pdb_file)
            if len(seq) < 2 or len(seq) > CFG.max_sequence_length:
                continue

            contact_map = compute_contact_map(coords)
            L = len(seq)

            tokens = tokenizer(seq, return_tensors="pt")
            input_ids = tokens["input_ids"].to(CFG.device)
            attention_mask = tokens["attention_mask"].to(CFG.device)

            with torch.no_grad():
                token_embeddings = esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.squeeze(0)
            token_embeddings = token_embeddings[1:L+1]

            idx_i, idx_j = torch.triu_indices(L, L, offset=1)
            delta_pos = (idx_j - idx_i).float().unsqueeze(1).to(CFG.device)

            logits = model(token_embeddings, idx_i.to(CFG.device), idx_j.to(CFG.device), delta_pos)
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            preds = (probs > 0.5).astype(int)
            labels = contact_map[idx_i.cpu(), idx_j.cpu()].astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)

            try:
                f1 = f1_score(labels, preds, average="binary")
                precision = precision_score(labels, preds)
                recall = recall_score(labels, preds)
                roc_auc = roc_auc_score(labels, probs)
                pr_auc = average_precision_score(labels, probs)
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            except:
                f1 = precision = recall = roc_auc = pr_auc = tn = fp = fn = tp = 0.0

            per_protein_records.append({
                "file": pdb_file,
                "seq_len": L,
                "num_pairs": len(labels),
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            })

        except Exception as e:
            logger.warning(f"Skipping {pdb_file}: {e}")

    if all_preds:
        global_f1 = f1_score(all_labels, all_preds)
        global_precision = precision_score(all_labels, all_preds)
        global_recall = recall_score(all_labels, all_preds)
        global_roc_auc = roc_auc_score(all_labels, all_probs)
        global_pr_auc = average_precision_score(all_labels, all_probs)
        try:
            gtn, gfp, gfn, gtp = confusion_matrix(all_labels, all_preds).ravel()
        except:
            gtn = gfp = gfn = gtp = 0

        logger.info(f"Test Set - F1: {global_f1:.4f}, Precision: {global_precision:.4f}, Recall: {global_recall:.4f},"
                    f" ROC-AUC: {global_roc_auc:.4f}, PR-AUC: {global_pr_auc:.4f}")

        summary_df = pd.DataFrame([{
            "F1 Score": global_f1,
            "Precision": global_precision,
            "Recall": global_recall,
            "ROC-AUC": global_roc_auc,
            "PR-AUC": global_pr_auc,
            "TN": gtn,
            "FP": gfp,
            "FN": gfn,
            "TP": gtp
        }])
        per_protein_df = pd.DataFrame(per_protein_records)

        output_path = CFG.reporting_dir.joinpath("test_set_evaluation.xlsx")
        with pd.ExcelWriter(output_path) as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            per_protein_df.to_excel(writer, sheet_name="per_protein", index=False)

        logger.success(f"Saved evaluation report to {output_path}")
    else:
        logger.warning("No valid predictions computed on test set.")

if __name__ == "__main__":
    evaluate_test_set()
