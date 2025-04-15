import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from tqdm import tqdm
from config import CFG
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel


class PairwiseContactModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, best_model_path=None):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim),  # +1 for delta_pos, +1 for prior map
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.best_model_path = best_model_path

    def forward(self, embeddings, idx1, idx2, delta_pos, similar_contact_prior=None):
        e1 = embeddings[idx1]
        e2 = embeddings[idx2]
        pairwise_reps = torch.cat([e1, e2, delta_pos.to(e1.device)], dim=-1)

        if similar_contact_prior is not None:
            pairwise_reps = torch.cat([pairwise_reps, similar_contact_prior.unsqueeze(1)], dim=-1)
        else:
            pairwise_reps = torch.cat(
                [pairwise_reps, torch.zeros((pairwise_reps.size(0), 1), device=pairwise_reps.device)], dim=-1)

        logits = self.classifier(pairwise_reps).squeeze(-1)
        return logits

    def infer(self, sequence: str, visualize: bool = False):
        if not self.best_model_path:
            logger.error("Provide best_model_path to load the model")
            return
        tokenizer = AutoTokenizer.from_pretrained(CFG.esm_model_name)
        esm_model = AutoModel.from_pretrained(CFG.esm_model_name).to(CFG.device)
        esm_model.eval()

        tokens = tokenizer(sequence, return_tensors="pt")
        input_ids = tokens["input_ids"].to(CFG.device)
        attention_mask = tokens["attention_mask"].to(CFG.device)

        with torch.no_grad():
            token_embeddings = esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.squeeze(0)

        L = token_embeddings.shape[0] - 2
        token_embeddings = token_embeddings[1:L+1]

        idx_i, idx_j = torch.triu_indices(L, L, offset=1)
        delta_pos = (idx_j - idx_i).float().unsqueeze(1).to(CFG.device)

        idx_i = idx_i.to(CFG.device)
        idx_j = idx_j.to(CFG.device)

        self.load_state_dict(torch.load(self.best_model_path, map_location=CFG.device))
        self.eval()

        with torch.no_grad():
            logits = self.forward(token_embeddings, idx_i, idx_j, delta_pos)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()

        contact_map = np.zeros((L, L), dtype=int)
        for i, j, val in zip(idx_i.cpu().numpy(), idx_j.cpu().numpy(), preds):
            contact_map[i, j] = val
            contact_map[j, i] = val

        if visualize:
            plt.figure(figsize=(6, 6))
            plt.imshow(contact_map, cmap='gray')
            plt.title("Predicted Contact Map")
            plt.xlabel("Residue Index")
            plt.ylabel("Residue Index")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        return contact_map

    def evaluate(self, loader, compute_extra_metrics: bool = False):
        self.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                try:
                    embeddings = batch["token_embeddings"].to(CFG.device)
                    idx1 = batch["idx1"].to(CFG.device)
                    idx2 = batch["idx2"].to(CFG.device)
                    delta_pos = batch["delta_pos"].to(CFG.device)
                    labels = batch["labels"].to(CFG.device)
                    similar_map = batch.get("most_similar_contact_map")
                    if similar_map is not None:
                        similar_map = similar_map.to(CFG.device)

                    logits = self.forward(embeddings, idx1, idx2, delta_pos, similar_contact_prior=similar_map)
                    probs = torch.sigmoid(logits)
                    preds = probs.cpu().numpy() > 0.5

                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

                    del embeddings, idx1, idx2, delta_pos, labels, logits, probs
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    logger.warning("Skipping batch during evaluation due to CUDA OOM.")
                    torch.cuda.empty_cache()
                    continue

        if not all_preds:
            return (0.0, 0.0, 0.0, 0.0) if compute_extra_metrics else (0.0, 0.0)

        logger.info("Calculating F1 score")
        f1 = f1_score(all_labels, all_preds)
        logger.info("Calculating precision")
        precision = precision_score(all_labels, all_preds)

        if compute_extra_metrics:
            logger.info("Calculating recall")
            recall = recall_score(all_labels, all_preds)
            logger.info("Calculating ROC-AUC")
            roc_auc = roc_auc_score(all_labels, all_probs)
            logger.info("Calculating PR-AUC")
            pr_auc = average_precision_score(all_labels, all_probs)
            logger.info(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
            return f1, precision, recall, roc_auc, pr_auc

        return f1, precision
