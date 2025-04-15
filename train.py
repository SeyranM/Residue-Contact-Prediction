import os
import json
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CFG
from data import PreprocessedTensorDataset
from loguru import logger
from model import PairwiseContactModel
from bucket_sampler import BucketBatchSampler


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return 1 - tversky_index


def collate_tensor_batches(batch):
    all_embeddings, all_idx1, all_idx2, all_labels, all_delta_pos = [], [], [], [], []
    prior_values = []
    offset = 0

    for item in batch:
        L = item["token_embeddings"].shape[0]
        all_embeddings.append(item["token_embeddings"])

        map_i = item.get("most_similar_contact_map")
        if map_i is not None:
            valid_mask = (item["idx1"] < map_i.size(0)) & (item["idx2"] < map_i.size(1))
            idx1_valid = item["idx1"][valid_mask]
            idx2_valid = item["idx2"][valid_mask]
            vals = torch.zeros_like(item["labels"])
            vals[valid_mask] = map_i[idx1_valid, idx2_valid]
        else:
            vals = torch.zeros_like(item["labels"])
        prior_values.append(vals)

        idx1 = item["idx1"] + offset
        idx2 = item["idx2"] + offset
        all_idx1.append(idx1)
        all_idx2.append(idx2)
        all_labels.append(item["labels"])
        all_delta_pos.append((item["idx2"] - item["idx1"]).float().unsqueeze(1))

        offset += L

    return {
        "token_embeddings": torch.cat(all_embeddings, dim=0),
        "idx1": torch.cat(all_idx1, dim=0),
        "idx2": torch.cat(all_idx2, dim=0),
        "delta_pos": torch.cat(all_delta_pos, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "most_similar_contact_prior": torch.cat(prior_values, dim=0)
    }


def train(resume_from=None):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    log_dir = CFG.logs_dir.joinpath(timestamp)
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = log_dir.joinpath("metrics.json")

    train_dataset = PreprocessedTensorDataset(str(CFG.processed_data_dir.joinpath("train")))
    val_dataset = PreprocessedTensorDataset(str(CFG.processed_data_dir.joinpath("val")))

    train_sampler = BucketBatchSampler(
        lengths=train_dataset.get_seq_lengths(),
        batch_size=CFG.train_batch_size,
        bucket_size=CFG.train_bucket_size,
        shuffle=True
    )
    val_sampler = BucketBatchSampler(
        lengths=val_dataset.get_seq_lengths(),
        batch_size=CFG.val_batch_size,
        bucket_size=CFG.val_bucket_size,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_tensor_batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_tensor_batches
    )

    model = PairwiseContactModel(embedding_dim=CFG.embedding_dim).to(CFG.device)

    if resume_from:
        logger.info(f"Resuming training using the weights file at location {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=CFG.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = TverskyLoss(alpha=0.7, beta=0.3)
    best_f1 = -1
    all_metrics = []

    for epoch in range(CFG.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CFG.epochs}"):
            embeddings = batch["token_embeddings"].to(CFG.device)
            idx1 = batch["idx1"].to(CFG.device)
            idx2 = batch["idx2"].to(CFG.device)
            delta_pos = batch["delta_pos"].to(CFG.device)
            labels = batch["labels"].to(CFG.device)
            prior = batch["most_similar_contact_prior"].to(CFG.device)

            logits = model(embeddings, idx1, idx2, delta_pos, similar_contact_prior=prior)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            del embeddings, idx1, idx2, delta_pos, labels, logits, prior
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        f1, precision, recall, roc_auc, pr_auc = model.evaluate(val_loader, compute_extra_metrics=True)

        logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, F1={f1:.4f}, Precision={precision:.4f}, "
                    f"Recall={recall:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            model_filename = f"contact_model_f1_{f1*100:.2f}_epoch_{best_epoch}.pt"
            model_path = CFG.models_dir.joinpath(model_filename)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path} with F1: {f1:.4f}")

        all_metrics.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)


if __name__ == '__main__':
    train(resume_from=CFG.best_model_path)
