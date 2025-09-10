from accelerate import Accelerator
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm.auto import tqdm
import torch
import numpy as np
from typing import Dict, Any

from .models.get_model import get_model_and_tokenizer
from .utils.metrics import compute_metrics

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model, self.tokenizer, self.device = get_model_and_tokenizer(
            config.get("model_checkpoint", "bert-base-cased")
        )
        
    def setup_training(self, train_dataset, validation_dataset):
        train_dataset.set_format("torch")
        validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
        validation_set.set_format("torch")

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.config.get("batch_size", 8),
        )
        self.eval_dataloader = DataLoader(
            validation_set, collate_fn=default_data_collator, batch_size=self.config.get("batch_size", 8)
        )

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.get("learning_rate", 2e-5))
        self.accelerator = Accelerator(mixed_precision=self.config.get("mixed_precision", "fp16"))
        
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

        num_train_epochs = self.config.get("num_epochs", 3)
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 0),
            num_training_steps=num_training_steps,
        )
        
        self.output_dir = self.config.get("output_dir", "bert-finetuned-squad-accelerate")
        self.progress_bar = tqdm(range(num_training_steps))
        
    def train_epoch(self, epoch: int):
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress_bar.update(1)
            
            if step % self.config.get("logging_steps", 100) == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    def evaluate(self, validation_dataset, raw_validation_dataset):
        self.model.eval()
        start_logits = []
        end_logits = []
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(**batch)

            start_logits.append(self.accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(self.accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        metrics = compute_metrics(
            start_logits, end_logits, validation_dataset, raw_validation_dataset
        )
        return metrics
    
    def save_model(self):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(self.output_dir, save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.output_dir)
    
    def train(self, train_dataset, validation_dataset, raw_validation_dataset):
        self.setup_training(train_dataset, validation_dataset)
        
        for epoch in range(self.config.get("num_epochs", 3)):
            print(f"Starting epoch {epoch + 1}")
            self.train_epoch(epoch)
            
            print("Running evaluation...")
            metrics = self.evaluate(validation_dataset, raw_validation_dataset)
            print(f"Epoch {epoch} metrics:", metrics)
            
            # Save checkpoint after each epoch
            self.save_model()
        
        print("Training completed!")