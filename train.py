import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

from model import create_model

class BloomDataset(Dataset):
    """
    Custom Dataset for Bloom's Taxonomy Classification
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Handle both numeric and categorical labels
        label_value = str(self.labels[idx])  # Convert to string first
        
        if label_value.startswith('L'):
            # Extract number from L1, L2, L3, etc.
            label = int(label_value[1:]) - 1  # L1->0, L2->1, L3->2, etc.
        else:
            # Handle numeric labels
            label = int(label_value) - 1  # Convert 1-6 to 0-5
        
        # Tokenize
        encoding = self.tokenizer.tokenize(
            text,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTTrainer:
    """
    Advanced BERT Training Pipeline
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create model and tokenizer
        self.model, self.tokenizer = create_model(model_name, num_classes)
        self.model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self, csv_path, test_size=0.2, val_size=0.5):
        """
        Load and split data
        Args:
            csv_path: path to CSV file
            test_size: proportion for test set
            val_size: proportion of remaining data for validation
        """
        print("Loading data...")
        df = pd.read_csv(csv_path)
        
        # Create numeric labels for stratification
        def convert_label_to_numeric(label):
            label_str = str(label)
            if label_str.startswith('L'):
                return int(label_str[1:])  # L1 -> 1, L2 -> 2, etc.
            else:
                return int(label_str)
        
        df['label_numeric'] = df['label'].apply(convert_label_to_numeric)
        
        # Verify data
        print(f"Total samples: {len(df)}")
        print(f"Classes distribution:")
        print(df['label'].value_counts().sort_index())
        print(f"Numeric label range: {df['label_numeric'].min()} to {df['label_numeric'].max()}")
        
        # Check label range
        if not df['label_numeric'].between(1, 6).all():
            print("Warning: Some labels are outside the expected range 1-6")
        
        # Split data stratified by numeric label
        texts = df['question'].values
        labels = df['label'].values  # Keep original labels (L1, L2, etc.)
        numeric_labels = df['label_numeric'].values  # Use for stratification
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test, num_temp, num_test = train_test_split(
            texts, labels, numeric_labels, test_size=test_size, 
            stratify=numeric_labels, random_state=42
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val, num_train, num_val = train_test_split(
            X_temp, y_temp, num_temp, test_size=val_size,
            stratify=num_temp, random_state=42
        )
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Show distribution
        print(f"Train label distribution: {pd.Series(y_train).value_counts().sort_index()}")
        print(f"Val label distribution: {pd.Series(y_val).value_counts().sort_index()}")
        
        # Create datasets
        self.train_dataset = BloomDataset(X_train, y_train, self.tokenizer)
        self.val_dataset = BloomDataset(X_val, y_val, self.tokenizer)
        self.test_dataset = BloomDataset(X_test, y_test, self.tokenizer)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(self, batch_size=8, num_workers=2):
        """Create optimized data loaders"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def setup_training(self, lr=3e-5, weight_decay=0.01, epochs=8):
        """Setup optimizer and scheduler with improved settings"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Calculate total steps based on actual epochs
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(0.15 * total_steps)  # 15% warmup
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.15,  # 15% warmup
            anneal_strategy='cos'
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        print(f"Scheduler: OneCycleLR (warmup_steps={warmup_steps})")
        print(f"Loss: CrossEntropyLoss with label_smoothing=0.1")
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Clear cache periodically to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def train(self, csv_path, epochs=8, batch_size=8, lr=3e-5, 
              patience=4, save_path='best_model.pth'):
        """
        Complete training pipeline with improved settings
        """
        print("=== BERT Base Bloom's Taxonomy Training (Optimized) ===")
        
        # Load data
        self.load_data(csv_path)
        
        # Create data loaders
        self.create_dataloaders(batch_size)
        
        # Setup training with actual epochs
        self.setup_training(lr, epochs=epochs)
        
        # Training loop - now using F1 score for best model selection
        best_val_f1 = 0
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}, Learning rate: {lr}")
        print(f"Model selection based on F1 score")
        print("-" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'epoch': epoch,
                    'model_name': self.model_name
                }, save_path)
                
                print(f"✅ New best model saved! F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Clear cache at end of epoch
            torch.cuda.empty_cache()
        
        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved to: {save_path}")
        
        return best_val_acc

def main():
    """Main training function with optimized settings"""
    # Ask for data path
    csv_path = input("Please enter the path to your CSV dataset file: ").strip()
    
    # Configuration - Optimized for BERT Base
    EPOCHS = 8          # Increased from 5
    BATCH_SIZE = 8      # Keep at 8 due to GPU limitation
    LEARNING_RATE = 3e-5  # Increased from 2e-5
    
    # Check if data file exists
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file '{csv_path}' not found!")
        print("Please make sure the file path is correct and the file exists.")
        return
    
    # Create trainer with BERT Base
    trainer = BERTTrainer(model_name='bert-base-uncased')
    
    # Train model
    best_accuracy = trainer.train(
        csv_path=csv_path,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    
    print(f"\n🎉 Training completed with best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()