import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class AdvancedBERTClassifier(nn.Module):
    """
    Advanced BERT Classifier for Bloom's Taxonomy Classification
    6-class classification (L1-L6) with production-level optimizations
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=6, dropout=0.3):
        super(AdvancedBERTClassifier, self).__init__()
        
        # BERT backbone
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from BERT config (768 for bert-base, 1024 for bert-large)
        hidden_size = self.bert.config.hidden_size
        
        # Advanced multi-layer classifier head
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 512)
        
        self.dropout2 = nn.Dropout(dropout * 0.7)  # Reduce dropout progressively
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.bn3 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in [self.fc1, self.fc2, self.classifier]:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        Args:
            input_ids: tokenized input sequences
            attention_mask: attention mask for padding
        Returns:
            logits: classification logits for 6 classes
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Multi-layer classifier with batch norm and dropout
        x = self.dropout1(pooled_output)
        x = self.bn1(x)
        x = F.gelu(self.fc1(x))  # GELU activation
        
        x = self.dropout2(x)
        x = self.bn2(x)
        x = F.gelu(self.fc2(x))
        
        x = self.dropout3(x)
        x = self.bn3(x)
        logits = self.classifier(x)
        
        return logits

class BERTTokenizer:
    """
    Tokenizer wrapper for BERT preprocessing
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize(self, texts, padding=True, truncation=True):
        """
        Tokenize input texts
        Args:
            texts: list of text strings or single string
            padding: whether to pad sequences
            truncation: whether to truncate long sequences
        Returns:
            tokenized inputs ready for model
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )

def create_model(model_name='bert-base-uncased', num_classes=6, dropout=0.3):
    """
    Factory function to create BERT classifier
    Args:
        model_name: pretrained model name (default: bert-base-uncased to match train.py)
        num_classes: number of output classes
        dropout: dropout rate
    Returns:
        model: initialized BERT classifier
        tokenizer: corresponding tokenizer
    """
    model = AdvancedBERTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout
    )
    tokenizer = BERTTokenizer(model_name=model_name)
    
    return model, tokenizer

def load_model(model_path, model_name='bert-base-uncased', num_classes=6):
    """
    Load trained model from checkpoint
    Args:
        model_path: path to saved model
        model_name: original model name (default: bert-base-uncased to match train.py)
        num_classes: number of classes
    Returns:
        model: loaded model
        tokenizer: corresponding tokenizer
    """
    model, tokenizer = create_model(model_name, num_classes)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

if __name__ == "__main__":
    # Test model creation
    model, tokenizer = create_model()
    
    # Test forward pass
    sample_text = ["What is photosynthesis?", "Analyze the themes in Romeo and Juliet"]
    inputs = tokenizer.tokenize(sample_text)
    
    model.eval()
    with torch.no_grad():
        logits = model(**inputs)
        predictions = torch.argmax(logits, dim=-1)
        print(f"Sample predictions: {predictions + 1}")  # Convert to 1-6 range
        print(f"Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Hidden size: {model.bert.config.hidden_size}")  # Show actual hidden size