import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# FIXED: Use proper imports from model.py
from model import create_model, load_model

class BloomDataset(Dataset):
    """Dataset class for Bloom's Taxonomy classification - FIXED to match train.py"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Handle both numeric and categorical labels (same as train.py)
        label_value = str(self.labels[idx])
        
        if label_value.startswith('L'):
            # Extract number from L1, L2, L3, etc.
            label = int(label_value[1:]) - 1  # L1->0, L2->1, L3->2, etc.
        else:
            # Handle numeric labels
            label = int(label_value) - 1  # Convert 1-6 to 0-5
        
        # FIXED: Use tokenizer from model.py (consistent with training)
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

def load_and_split_data(csv_file):
    """Load CSV data and create train/val/test splits"""
    print("Loading dataset...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Dataset loaded: {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return None, None, None
    
    # Verify data structure
    if 'question' not in df.columns or 'label' not in df.columns:
        print("Error: CSV must have 'question' and 'label' columns")
        return None, None, None
    
    # Print class distribution
    print("\nClass distribution:")
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"L{label}: {count} samples")
    
    # Create numeric labels for stratification (same as train.py)
    def convert_label_to_numeric(label):
        label_str = str(label)
        if label_str.startswith('L'):
            return int(label_str[1:])  # L1 -> 1, L2 -> 2, etc.
        else:
            return int(label_str)
    
    df['label_numeric'] = df['label'].apply(convert_label_to_numeric)
    
    # Split data using same logic as train.py
    texts = df['question'].values
    labels = df['label'].values  # Keep original labels
    numeric_labels = df['label_numeric'].values  # Use for stratification
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test, num_temp, num_test = train_test_split(
        texts, labels, numeric_labels, test_size=0.2, 
        stratify=numeric_labels, random_state=42
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val, num_train, num_val = train_test_split(
        X_temp, y_temp, num_temp, test_size=0.5,
        stratify=num_temp, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create DataFrames
    train_df = pd.DataFrame({'question': X_train, 'label': y_train})
    val_df = pd.DataFrame({'question': X_val, 'label': y_val})
    test_df = pd.DataFrame({'question': X_test, 'label': y_test})
    
    return train_df, val_df, test_df

def evaluate_model(model, dataloader, device, dataset_name="Dataset"):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    print(f"\nEvaluating on {dataset_name}...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # FIXED: use 'labels' not 'label'
            
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert back to 1-6 labels for reporting
    all_predictions = [p + 1 for p in all_predictions]
    all_labels = [l + 1 for l in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=[1,2,3,4,5,6]
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1,
        'support': support,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels
    }

def print_metrics(metrics, dataset_name):
    """Print comprehensive evaluation metrics"""
    print(f"\n{'='*50}")
    print(f"{dataset_name.upper()} RESULTS")
    print(f"{'='*50}")
    
    print(f"Overall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-Macro:  {metrics['f1_macro']:.4f}")
    print(f"  F1-Micro:  {metrics['f1_micro']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
    
    print(f"{'Level':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 62)
    
    for i, level in enumerate(bloom_levels):
        print(f"L{i+1}-{level:<10} {metrics['precision'][i]:.4f}     {metrics['recall'][i]:.4f}     {metrics['f1_scores'][i]:.4f}     {metrics['support'][i]:<10}")
    
    print("-" * 62)
    print(f"{'Macro Avg':<12} {np.mean(metrics['precision']):.4f}     {np.mean(metrics['recall']):.4f}     {metrics['f1_macro']:.4f}     {sum(metrics['support']):<10}")

def plot_confusion_matrix(y_true, y_pred, dataset_name, save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5,6])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['L1-Remember', 'L2-Understand', 'L3-Apply', 'L4-Analyze', 'L5-Evaluate', 'L6-Create'],
                yticklabels=['L1-Remember', 'L2-Understand', 'L3-Apply', 'L4-Analyze', 'L5-Evaluate', 'L6-Create'])
    
    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def main():
    """Main evaluation function"""
    print("BERT Bloom's Taxonomy Classification - Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get CSV file path
    csv_path = input("Please enter the path to your CSV dataset file: ").strip()
    
    # Load data
    train_df, val_df, test_df = load_and_split_data(csv_path)
    if train_df is None:
        return
    
    # FIXED: Load model using proper factory functions
    try:
        print("\nLoading trained model...")
        model, tokenizer = load_model('best_model.pth', model_name='bert-base-uncased')
        model.to(device)
        print("Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except FileNotFoundError:
        print("Error: best_model.pth not found! Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create datasets using consistent tokenizer
    val_dataset = BloomDataset(val_df['question'].values, val_df['label'].values, tokenizer)
    test_dataset = BloomDataset(test_df['question'].values, test_df['label'].values, tokenizer)
    
    # Create dataloaders
    batch_size = 16  # Slightly larger for evaluation
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_dataloader, device, "Validation Set")
    print_metrics(val_metrics, "Validation")
    
    # Evaluate on test set  
    test_metrics = evaluate_model(model, test_dataloader, device, "Test Set")
    print_metrics(test_metrics, "Test")
    
    # Plot confusion matrices
    plot_confusion_matrix(val_metrics['labels'], val_metrics['predictions'], 
                         "Validation Set", "confusion_matrix_val.png")
    
    plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], 
                         "Test Set", "confusion_matrix_test.png")
    
    # Print classification reports
    print(f"\n{'='*50}")
    print("DETAILED CLASSIFICATION REPORT - VALIDATION")
    print(f"{'='*50}")
    bloom_labels = ['L1-Remember', 'L2-Understand', 'L3-Apply', 'L4-Analyze', 'L5-Evaluate', 'L6-Create']
    print(classification_report(val_metrics['labels'], val_metrics['predictions'], 
                              target_names=bloom_labels, digits=4))
    
    print(f"\n{'='*50}")
    print("DETAILED CLASSIFICATION REPORT - TEST")
    print(f"{'='*50}")
    print(classification_report(test_metrics['labels'], test_metrics['predictions'], 
                              target_names=bloom_labels, digits=4))
    
    # Save evaluation results
    results = {
        'validation': {
            'accuracy': float(val_metrics['accuracy']),
            'f1_macro': float(val_metrics['f1_macro']),
            'f1_micro': float(val_metrics['f1_micro']),
            'loss': float(val_metrics['loss'])
        },
        'test': {
            'accuracy': float(test_metrics['accuracy']),
            'f1_macro': float(test_metrics['f1_macro']),
            'f1_micro': float(test_metrics['f1_micro']),
            'loss': float(test_metrics['loss'])
        }
    }
    
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to evaluation_results.json")
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1-Macro: {test_metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    main()