from loguru import logger
from preprocessing import ContactMapPreprocessor, EmbeddingPreprocessor
from train import train


def run_preprocessing():
    logger.info("Running ContactMapPreprocessor...")
    ContactMapPreprocessor(n_processes=8).run()
    logger.success("ContactMapPreprocessor completed.")

    logger.info("Running EmbeddingPreprocessor...")
    embedding_preprocessor = EmbeddingPreprocessor(batch_size=2)
    embedding_preprocessor.run()
    embedding_preprocessor.add_similarity_priors("train")
    embedding_preprocessor.add_similarity_priors("val")
    logger.success("EmbeddingPreprocessor completed.")


def run_training():
    logger.info("Starting training...")
    train()
    logger.success("Training completed successfully.")


def run_pipeline():
    run_preprocessing()
    run_training()


if __name__ == "__main__":
    run_pipeline()
