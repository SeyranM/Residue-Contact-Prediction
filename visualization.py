import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch
from collections import Counter, defaultdict
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from loguru import logger
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser

from config import CFG


def plot_metrics():
    log_dir = CFG.logs_dir
    metrics_path = log_dir.joinpath("metrics.json")
    if not metrics_path.exists():
        logger.error(f"No metrics.json found in {log_dir}")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    f1_scores = data.get("f1", [])
    precision_scores = data.get("precision", [])
    epochs = list(range(1, len(f1_scores) + 1))

    plt.figure()
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.plot(epochs, precision_scores, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Validation Metrics - {os.path.basename(log_dir)}')
    plt.legend()
    plt.grid(True)

    plot_path = CFG.visualization_dir.joinpath("val_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    logger.success(f"Saved metric plot to {plot_path}")


def plot_sequence_length_distribution():
    lengths_path = CFG.processed_data_dir.joinpath("sequence_lengths.csv")
    if not lengths_path.exists():
        logger.error(f"sequence_lengths.csv not found in {CFG.processed_data_dir}")
        return

    df = pd.read_csv(lengths_path)
    lengths = df['length'].tolist()

    output_path = CFG.visualization_dir.joinpath("seq_length_dist.png")
    plt.figure()
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Sequence Length Distribution from Processed Data')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logger.success(f"Saved sequence length distribution to {output_path}")


def count_labels(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')
        labels = data["labels"].tolist()
        return Counter(labels)
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return Counter()


def plot_label_distribution():
    processed_data_dir = CFG.processed_data_dir
    for split in ["train", "val"]:
        processed_data_split_dir = processed_data_dir.joinpath(split)
        output_path = CFG.visualization_dir.joinpath(f"{split}_label_distribution.png")
        pt_files = [os.path.join(processed_data_split_dir, f) for f in os.listdir(processed_data_split_dir) if
                    f.endswith(".pt")]

        label_counts = Counter()
        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap(count_labels, pt_files), total=len(pt_files),
                               desc=f"{split} - Label Counting"):
                label_counts.update(result)

        labels = list(label_counts.keys())
        counts = list(label_counts.values())

        plt.figure()
        plt.bar(labels, counts, color='coral', edgecolor='black')
        plt.xlabel('Label Value')
        plt.ylabel('Count')
        plt.title('Label Distribution in Preprocessed Data')
        plt.xticks([0, 1])
        plt.grid(True, axis='y')
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Saved label distribution to {output_path}")


def compute_density(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')
        L = data["token_embeddings"].shape[0]
        contact_count = int(sum(data["labels"]))
        density = contact_count / (L * (L - 1) / 2) if L > 1 else 0
        return L, density
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None


def plot_contact_density_vs_length():
    processed_data_dir = CFG.processed_data_dir
    for split in ["train", "val"]:
        processed_data_split_dir = processed_data_dir.joinpath(split)
        output_path = CFG.visualization_dir.joinpath(f"{split}_contact_density.png")
        pt_files = [os.path.join(processed_data_split_dir, f) for f in os.listdir(processed_data_split_dir) if
                    f.endswith(".pt")]

        lengths = []
        densities = []
        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap(compute_density, pt_files), total=len(pt_files),
                               desc=f"{split} - Density Calc"):
                if result is not None:
                    L, density = result
                    lengths.append(L)
                    densities.append(density)

        plt.figure()
        plt.scatter(lengths, densities, alpha=0.5)
        plt.xlabel('Sequence Length')
        plt.ylabel('Contact Density')
        plt.title('Contact Density vs. Sequence Length')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Saved contact density plot to {output_path}")


def count_residue_contacts(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')
        idx1 = data["idx1"].tolist()
        idx2 = data["idx2"].tolist()
        labels = data["labels"].tolist()
        count_dict = Counter()
        for i, (a, b) in enumerate(zip(idx1, idx2)):
            if labels[i] > 0.5:
                count_dict[a] += 1
                count_dict[b] += 1
        return list(count_dict.values())
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return []


def plot_contacts_per_residue():
    processed_data_dir = CFG.processed_data_dir
    for split in ["train", "val"]:
        processed_data_split_dir = processed_data_dir.joinpath(split)
        output_path = CFG.visualization_dir.joinpath(f"{split}_contacts_per_residue.png")
        pt_files = [os.path.join(processed_data_split_dir, f) for f in os.listdir(processed_data_split_dir) if
                    f.endswith(".pt")]

        contact_counts = []
        with Pool(processes=cpu_count()) as pool:
            for counts in tqdm(pool.imap(count_residue_contacts, pt_files), total=len(pt_files),
                               desc=f"{split} - Per-residue Count"):
                contact_counts.extend(counts)

        plt.figure()
        plt.hist(contact_counts, bins=50, color='purple', edgecolor='black')
        plt.xlabel('Number of Contacts')
        plt.ylabel('Residue Count')
        plt.title('Number of Contacts per Residue')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Saved contacts per residue histogram to {output_path}")


def visualize_prior(file_name: str):
    train_path = CFG.processed_data_dir.joinpath("train", f"{file_name}.pt")
    if not train_path.exists():
        logger.error(f"File {train_path} does not exist.")
        return

    data = torch.load(train_path, map_location="cpu")
    contact_map = torch.zeros((data["token_embeddings"].shape[0], data["token_embeddings"].shape[0]))

    for (i, j), label in zip(zip(data["idx1"], data["idx2"]), data["labels"]):
        contact_map[i, j] = label
        contact_map[j, i] = label

    similar_map = data.get("most_similar_contact_map")
    if similar_map is None:
        logger.warning(f"No similar contact map found in {file_name}.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(contact_map.numpy(), cmap="gray")
    axs[0].set_title("Ground Truth Contact Map")
    axs[0].set_xlabel("Residue Index")
    axs[0].set_ylabel("Residue Index")

    axs[1].imshow(similar_map.numpy(), cmap="gray")
    axs[1].set_title("Most Similar Structural Prior")
    axs[1].set_xlabel("Residue Index")
    axs[1].set_ylabel("Residue Index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # if not CFG.visualization_dir.exists():
    #     os.makedirs(CFG.visualization_dir, exist_ok=True)
    #
    # plot_metrics()
    # plot_sequence_length_distribution()
    #
    # if CFG.processed_data_dir.exists():
    #     plot_label_distribution()
    #     plot_contact_density_vs_length()
    #     plot_contacts_per_residue()
    visualize_prior(CFG.processed_data_dir.joinpath("train", "1A2G"))
