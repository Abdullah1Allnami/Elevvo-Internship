from datasets import load_dataset
from src.preprocessing import DataPreprocessor
from src.train import Trainer
from tests.evaluate import evaluate_model_on_sample, batch_evaluate

def main():
    # Load dataset
    print("Loading dataset...")
    raw_datasets = load_dataset("squad")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(model_checkpoint="bert-base-cased")
    processed_datasets = preprocessor.prepare_datasets(raw_datasets)
    
    # Training configuration
    config = {
        "model_checkpoint": "bert-base-cased",
        "batch_size": 8,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "output_dir": "bert-finetuned-squad-accelerate",
        "mixed_precision": "fp16",
        "warmup_steps": 0,
        "logging_steps": 100
    }
    
    # Train model
    print("Starting training...")
    trainer = Trainer(config)
    trainer.train(
        processed_datasets["train"],
        processed_datasets["validation"],
        raw_datasets["validation"]
    )
    
    # Evaluate on a few samples
    print("\nEvaluating on sample data...")
    batch_evaluate(config["output_dir"], raw_datasets["validation"], num_samples=3)

if __name__ == "__main__":
    main()