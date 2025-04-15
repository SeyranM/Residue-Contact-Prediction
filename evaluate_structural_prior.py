import os
import torch
import pandas as pd
from tqdm import tqdm
from config import CFG
from loguru import logger
from multiprocessing import Pool, cpu_count

def check_prior(path):
    try:
        data = torch.load(path, map_location="cpu")
        file_id = os.path.basename(path).replace(".pt", "")
        has_prior = int(data.get("most_similar_contact_map") is not None)
        return {"file": file_id, "has_prior": has_prior}
    except Exception as e:
        logger.warning(f"Failed to process {path}: {e}")
        return None

def evaluate_structural_priors():
    for split in ["train", "val"]:
        processing_dir = CFG.processed_data_dir.joinpath(split)
        pt_files = [os.path.join(processing_dir, f) for f in os.listdir(processing_dir) if f.endswith(".pt")]

        results = []
        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap(check_prior, pt_files), total=len(pt_files), desc=f"Checking priors - {split}"):
                if result is not None:
                    results.append(result)

        df = pd.DataFrame(results)
        output_path = CFG.reporting_dir.joinpath(f"{split}_structural_prior_presence.xlsx")
        df.to_excel(output_path, index=False)
        logger.success(f"Saved structural prior presence data to {output_path}")


if __name__ == '__main__':
    evaluate_structural_priors()