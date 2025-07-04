import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import datasets
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import wandb
from collections import defaultdict
import matplotlib.pyplot as plt
import glob

# Import our model
from model.wednesday_mark2 import Wednesday, WednesdayConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """WikiText dataset for language modeling"""

    def __init__(self,
                 texts: List[str],
                 tokenizer,
                 max_length: int = 512,
                 split_name: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_name = split_name

        # Tokenize all texts
        self.tokenized_texts = []
        logger.info(f"Tokenizing {len(texts)} texts for {split_name}...")

        for text in tqdm(texts, desc=f"Tokenizing {split_name}"):
            if len(text.strip()) > 0:  # Skip empty texts
                tokens = self.tokenizer.encode(text, add_special_tokens=True)

                # Split long texts into chunks
                for i in range(0, len(tokens), max_length):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) > 10:  # Skip very short chunks
                        self.tokenized_texts.append(chunk)

        logger.info(
            f"Created {len(self.tokenized_texts)} samples for {split_name}")

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]

        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id
                               ] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def load_shakesphere(tokenizer,
                     max_length: int = 512
                     ) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and prepare tiny shakesphere dataset"""
    logger.info("Loading tiny shakesphere dataset...")

    # Load dataset
    dataset = datasets.load_dataset("text",
                                    data_files="./data/shakesphere/input.txt")

    texts = [item['text'] for item in dataset['train']]  # type: ignore
    # Create splits: 80% train, 10% val, 10% test
    total = len(texts)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    # Create datasets
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]

    # Create CustomDataset instances
    train_dataset = CustomDataset(train_texts, tokenizer, max_length, "train")
    val_dataset = CustomDataset(val_texts, tokenizer, max_length, "validation")
    test_dataset = CustomDataset(test_texts, tokenizer, max_length, "test")

    return train_dataset, val_dataset, test_dataset


class TrainingConfig:
    """Configuration for training"""

    def __init__(self):
        # Model hyperparameters
        self.vocab_size = 50257  # GPT-2 tokenizer
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.n_experts = 8
        self.top_k = 2
        self.max_seq_len = 512
        self.dropout = 0.1

        # Training hyperparameters
        self.batch_size = 16
        self.learning_rate = 1e-2
        self.weight_decay = 0.01
        self.num_epochs = 10
        self.warmup_steps = 1000
        self.gradient_accumulation_steps = 2
        self.max_grad_norm = 1.0
        # Learning rate decay
        self.decay_lr = True
        self.min_lr = 1e-4
        self.lr_decay_iters = 10000  # Set to total training steps or as desired

        # Evaluation
        self.eval_every = 500
        self.save_every = 1000
        self.generate_every = 1000

        # Paths
        self.output_dir = "outputs"
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"

        # Logging
        self.use_wandb = False
        self.wandb_project = "gpt-moe"
        self.wandb_run_name = None

        # Device
        self.device = "cuda" if torch.cuda.is_available(
        ) else "mps" if torch.mps.is_available() else "cpu"

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items() if not k.startswith('_')
        }


class GPTMoETrainer:
    """Trainer for GPT-MoE model"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.train_dataset, self.val_dataset, self.test_dataset = load_shakesphere(
            self.tokenizer, config.max_seq_len)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True)

        # Initialize model
        self.model_config = WednesdayConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layer=config.n_layers,
            n_experts=config.n_experts,
            top_k=config.top_k,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pad_token_id=self.tokenizer.pad_token_id)

        self.model = Wednesday(self.model_config).to(self.device)

        logger.info(f"Model has {self.model.count_parameters():,} parameters")

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)

        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(project=config.wandb_project,
                       name=config.wandb_run_name,
                       config=config.to_dict())

    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save model checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.step}.pt"

        checkpoint_path = os.path.join(self.config.checkpoint_dir,
                                       checkpoint_name)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'training_history': dict(self.training_history)
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save model in HuggingFace format
        model_save_path = os.path.join(self.config.output_dir,
                                       f"model_step_{self.step}")
        self.model.save_pretrained(model_save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = defaultdict(list,
                                            checkpoint['training_history'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given dataloader"""
        self.model.eval()

        total_loss = 0
        total_load_balance_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch['input_ids'],
                                     batch['attention_mask'], batch['labels'])

                loss = outputs['loss']
                load_balance_loss = outputs['load_balance_loss']

                # Count non-padding tokens
                mask = batch['attention_mask']
                num_tokens = mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_load_balance_loss += load_balance_loss.item(
                ) * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        avg_load_balance_loss = total_load_balance_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'loss': avg_loss,
            'load_balance_loss': avg_load_balance_loss,
            'perplexity': perplexity
        }

    def generate_sample(self,
                        prompt: str = "The future of artificial intelligence",
                        max_length: int = 100) -> str:
        """Generate sample text for monitoring training progress"""
        self.model.eval()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt,
                                          return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids,
                                                max_length=max_length,
                                                temperature=0.8,
                                                top_k=50,
                                                top_p=0.9)

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0],
                                               skip_special_tokens=True)
        return generated_text

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Forward pass
        outputs = self.model(batch['input_ids'], batch['attention_mask'],
                             batch['labels'])

        loss = outputs['loss']
        load_balance_loss = outputs['load_balance_loss']

        # Backward pass
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'load_balance_loss': load_balance_loss.item()
        }

    def get_lr(self, step):
        cfg = self.config
        if not cfg.decay_lr:
            return cfg.learning_rate
        # 1) linear warmup for warmup_steps steps
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / (cfg.warmup_steps + 1)
        # 2) if step > lr_decay_iters, return min learning rate
        if step > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - cfg.warmup_steps) / (cfg.lr_decay_iters -
                                                   cfg.warmup_steps)
        decay_ratio = min(max(decay_ratio, 0), 1)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        # Training loop
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_losses = []

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Training step
                step_metrics = self.train_step(batch)
                epoch_losses.append(step_metrics['loss'])

                # Gradient accumulation
                if (batch_idx +
                        1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.config.max_grad_norm)

                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update learning rate using custom schedule
                    lr = self.get_lr(self.step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                    self.step += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss':
                        f"{step_metrics['loss']:.4f}",
                        'lr':
                        f"{self.get_lr(self.step):.6f}"
                    })

                    # Log metrics
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss':
                            step_metrics['loss'],
                            'train/load_balance_loss':
                            step_metrics['load_balance_loss'],
                            'train/learning_rate':
                            self.get_lr(self.step),
                            'train/step':
                            self.step
                        })

                    # Store training history
                    self.training_history['train_loss'].append(
                        step_metrics['loss'])
                    self.training_history['train_load_balance_loss'].append(
                        step_metrics['load_balance_loss'])
                    self.training_history['learning_rate'].append(
                        self.get_lr(self.step))

                    # Evaluation
                    if self.step % self.config.eval_every == 0:
                        logger.info(f"Evaluating at step {self.step}...")
                        val_metrics = self.evaluate(self.val_loader)

                        logger.info(
                            f"Validation - Loss: {val_metrics['loss']:.4f}, "
                            f"Perplexity: {val_metrics['perplexity']:.2f}")

                        # Store validation history
                        self.training_history['val_loss'].append(
                            val_metrics['loss'])
                        self.training_history['val_perplexity'].append(
                            val_metrics['perplexity'])

                        # Log to wandb
                        if self.config.use_wandb:
                            wandb.log({
                                'val/loss':
                                val_metrics['loss'],
                                'val/perplexity':
                                val_metrics['perplexity'],
                                'val/load_balance_loss':
                                val_metrics['load_balance_loss']
                            })

                        # Save best model
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                            self.save_checkpoint("best_model.pt")
                            logger.info(
                                f"New best model saved with validation loss: {self.best_val_loss:.4f}"
                            )

                    # Generate sample
                    if self.step % self.config.generate_every == 0:
                        sample_text = self.generate_sample()
                        logger.info(f"Generated sample:\n{sample_text}")

                        if self.config.use_wandb:
                            wandb.log({"generated_text": sample_text})

                    # Save checkpoint
                    if self.step % self.config.save_every == 0:
                        self.save_checkpoint()

            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            logger.info(
                f"Epoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}"
            )
            self.epoch += 1

        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        final_val_metrics = self.evaluate(self.val_loader)
        final_test_metrics = self.evaluate(self.test_loader)

        logger.info(
            f"Final Validation - Loss: {final_val_metrics['loss']:.4f}, "
            f"Perplexity: {final_val_metrics['perplexity']:.2f}")
        logger.info(f"Final Test - Loss: {final_test_metrics['loss']:.4f}, "
                    f"Perplexity: {final_test_metrics['perplexity']:.2f}")

        # Save final model
        self.save_checkpoint("final_model.pt")

        # Save training history
        history_path = os.path.join(self.config.log_dir,
                                    "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(dict(self.training_history), f, indent=2)

        return final_test_metrics

    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        axes[0, 0].plot(self.training_history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')

        # Validation loss
        if 'val_loss' in self.training_history:
            val_steps = np.arange(0, len(
                self.training_history['val_loss'])) * self.config.eval_every
            axes[0, 1].plot(val_steps, self.training_history['val_loss'])
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')

        # Learning rate
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('LR')

        # Perplexity
        if 'val_perplexity' in self.training_history:
            val_steps = np.arange(0,
                                  len(self.training_history['val_perplexity'])
                                  ) * self.config.eval_every
            axes[1, 1].plot(val_steps, self.training_history['val_perplexity'])
            axes[1, 1].set_title('Validation Perplexity')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Perplexity')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_curves.png'))
        plt.show()


def main():
    """Main training function"""
    # Initialize config
    config = TrainingConfig()

    # Override config if needed
    config.batch_size = 8  # Reduce if OOM
    config.num_epochs = 5
    # config.eval_every = 200
    config.eval_every = 5
    # config.save_every = 500
    config.save_every = 10
    config.generate_every = 500

    # Initialize trainer
    trainer = GPTMoETrainer(config)

    # --- Resume logic ---
    checkpoint_dir = config.checkpoint_dir
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    latest_checkpoint = None
    if checkpoint_files:
        # Sort by step number (extracted from filename)
        def extract_step(path):
            import re
            match = re.search(r'checkpoint_step_(\\d+).pt',
                              os.path.basename(path))
            return int(match.group(1)) if match else -1

        checkpoint_files.sort(key=extract_step, reverse=True)
        latest_checkpoint = checkpoint_files[0]

    if latest_checkpoint:
        logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    # Train model
    final_metrics = trainer.train()

    # Plot training curves
    trainer.plot_training_curves()

    print(f"Training completed!")
    print(f"Final test metrics: {final_metrics}")


if __name__ == "__main__":
    main()
