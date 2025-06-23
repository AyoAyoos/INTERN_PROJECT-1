import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import sys
import os
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# For Excel support
try:
    import openpyxl
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

from model import AdvancedBERTClassifier

class BloomPredictor:
    """BERT-based Bloom's Taxonomy level predictor"""
    
    def __init__(self, model_path='best_model.pth', device=None):
        """Initialize the predictor with trained model"""
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model first to get model_name from checkpoint
        print("Loading trained model...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model name from checkpoint, default to bert-base-uncased
            model_name = checkpoint.get('model_name', 'bert-base-uncased')
            print(f"Model type: {model_name}")
            
            # Load tokenizer with correct model name
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model with correct model name
            self.model = AdvancedBERTClassifier(
                model_name=model_name,
                num_classes=6
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            
        except FileNotFoundError:
            print(f"Error: {model_path} not found! Please train the model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Bloom's taxonomy level descriptions
        self.bloom_levels = {
            1: "Remember - Recall facts and basic concepts",
            2: "Understand - Explain ideas or concepts", 
            3: "Apply - Use information in new situations",
            4: "Analyze - Draw connections among ideas",
            5: "Evaluate - Justify a stand or decision",
            6: "Create - Produce new or original work"
        }
    
    def preprocess_text(self, text, max_length=512):
        """Preprocess text for model input"""
        
        # Handle empty or None input
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        text = str(text).strip()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_single(self, text, return_probabilities=False, return_confidence=False):
        """
        Predict Bloom's taxonomy level for a single text
        
        Args:
            text (str): Input question/text to classify
            return_probabilities (bool): Whether to return class probabilities
            return_confidence (bool): Whether to return prediction confidence
            
        Returns:
            dict: Prediction results containing level, description, and optionally probabilities/confidence
        """
        
        try:
            # Preprocess text
            inputs = self.preprocess_text(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Convert to 1-6 label
            predicted_level = predicted_class + 1
            
            # Prepare result
            result = {
                'level': predicted_level,
                'description': self.bloom_levels[predicted_level],
                'text': text
            }
            
            if return_probabilities:
                prob_dict = {}
                for i, prob in enumerate(probabilities[0]):
                    level = i + 1
                    prob_dict[f"L{level}"] = float(prob.item())
                result['probabilities'] = prob_dict
            
            if return_confidence:
                result['confidence'] = float(confidence)
            
            return result
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'text': text
            }
    
    def predict_batch(self, texts, batch_size=32):
        """
        Predict Bloom's taxonomy levels for multiple texts
        
        Args:
            texts (list): List of input texts to classify
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            try:
                # Prepare batch inputs
                batch_inputs = []
                valid_texts = []
                
                for text in batch_texts:
                    if text and str(text).strip():
                        try:
                            inputs = self.preprocess_text(text)
                            batch_inputs.append(inputs)
                            valid_texts.append(text)
                        except Exception as e:
                            batch_results.append({
                                'error': f'Preprocessing failed: {str(e)}',
                                'text': text,
                                'level': None,
                                'description': None,
                                'confidence': None
                            })
                    else:
                        batch_results.append({
                            'error': 'Empty text',
                            'text': text,
                            'level': None,
                            'description': None,
                            'confidence': None
                        })
                
                if not batch_inputs:
                    results.extend(batch_results)
                    continue
                
                # Stack tensors
                input_ids = torch.cat([inp['input_ids'] for inp in batch_inputs], dim=0)
                attention_mask = torch.cat([inp['attention_mask'] for inp in batch_inputs], dim=0)
                
                # Make predictions
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_classes = torch.argmax(outputs, dim=1)
                    confidences = torch.max(probabilities, dim=1)[0]
                
                # Process results
                for j, text in enumerate(valid_texts):
                    predicted_level = predicted_classes[j].item() + 1
                    confidence = confidences[j].item()
                    
                    batch_results.append({
                        'text': text,
                        'level': predicted_level,
                        'description': self.bloom_levels[predicted_level],
                        'confidence': float(confidence),
                        'error': None
                    })
                
                results.extend(batch_results)
                
            except Exception as e:
                # Handle batch errors
                for text in batch_texts:
                    results.append({
                        'error': f"Batch prediction failed: {str(e)}",
                        'text': text,
                        'level': None,
                        'description': None,
                        'confidence': None
                    })
        
        return results
    
    def predict_from_file(self, file_path, output_path=None, question_column='question', sheet_name=0):
        """
        Predict Bloom's taxonomy levels for questions in a CSV or Excel file
        
        Args:
            file_path (str): Path to input CSV or Excel file
            output_path (str): Path to save output file (optional)
            question_column (str): Name of the column containing questions
            sheet_name (str/int): Sheet name or index for Excel files (default: 0)
            
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        
        try:
            # Determine file type and read accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                print(f"Reading CSV file: {file_path}")
                df = pd.read_csv(file_path)
                
            elif file_ext in ['.xlsx', '.xls']:
                if not EXCEL_SUPPORT:
                    raise ImportError("Excel support not available. Please install openpyxl: pip install openpyxl")
                
                print(f"Reading Excel file: {file_path}")
                print(f"Sheet: {sheet_name}")
                
                # Read Excel file
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
            else:
                raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .csv, .xlsx, .xls")
            
            # Check if question column exists
            if question_column not in df.columns:
                available_cols = ', '.join(df.columns.tolist())
                raise ValueError(f"Column '{question_column}' not found. Available columns: {available_cols}")
            
            print(f"Found {len(df)} questions to classify")
            
            # Get questions
            questions = df[question_column].tolist()
            
            # Make predictions
            print("Making predictions...")
            predictions = self.predict_batch(questions, batch_size=16)
            
            # Create results DataFrame
            results_df = df.copy()
            
            # Add prediction columns
            results_df['predicted_level'] = [p.get('level') for p in predictions]
            results_df['predicted_description'] = [p.get('description') for p in predictions]
            results_df['confidence'] = [p.get('confidence') for p in predictions]
            results_df['prediction_error'] = [p.get('error') for p in predictions]
            
            # Save if output path provided
            if output_path:
                # Determine output format based on extension
                output_ext = os.path.splitext(output_path)[1].lower()
                
                if output_ext == '.csv':
                    results_df.to_csv(output_path, index=False)
                elif output_ext in ['.xlsx', '.xls']:
                    if not EXCEL_SUPPORT:
                        print("Warning: Excel support not available. Saving as CSV instead.")
                        output_path = output_path.rsplit('.', 1)[0] + '.csv'
                        results_df.to_csv(output_path, index=False)
                    else:
                        results_df.to_excel(output_path, index=False)
                else:
                    # Default to CSV if extension not recognized
                    output_path = output_path + '.csv'
                    results_df.to_csv(output_path, index=False)
                
                print(f"Results saved to: {output_path}")
            
            # Print summary
            successful_predictions = sum(1 for p in predictions if p.get('level') is not None)
            print(f"\nPrediction Summary:")
            print(f"Total questions: {len(predictions)}")
            print(f"Successful predictions: {successful_predictions}")
            print(f"Failed predictions: {len(predictions) - successful_predictions}")
            
            if successful_predictions > 0:
                level_counts = results_df['predicted_level'].value_counts().sort_index()
                print(f"\nPredicted Level Distribution:")
                for level, count in level_counts.items():
                    if pd.notna(level):
                        print(f"  L{int(level)}: {count} questions")
            
            return results_df
            
        except Exception as e:
            print(f"Error processing file: {e}")
            raise
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n" + "="*60)
        print("BERT Bloom's Taxonomy Interactive Predictor")
        print("="*60)
        print("Enter questions to classify (type 'quit' to exit)")
        print("\nBloom's Taxonomy Levels:")
        for level, desc in self.bloom_levels.items():
            print(f"  L{level}: {desc}")
        print("\n" + "-"*60)
        
        while True:
            try:
                text = input("\nEnter question: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not text:
                    print("Please enter a valid question.")
                    continue
                
                # Make prediction
                result = self.predict_single(text, return_confidence=True, return_probabilities=True)
                
                if 'error' in result and result['error']:
                    print(f"Error: {result['error']}")
                    continue
                
                # Display results
                print(f"\n{'='*50}")
                print(f"Question: {result['text']}")
                print(f"{'='*50}")
                print(f"Predicted Level: L{result['level']}")
                print(f"Description: {result['description']}")
                print(f"Confidence: {result['confidence']:.4f}")
                
                if 'probabilities' in result:
                    print(f"\nAll Level Probabilities:")
                    for level_key, prob in result['probabilities'].items():
                        print(f"  {level_key}: {prob:.4f}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(
        description='BERT Bloom\'s Taxonomy Level Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question prediction
  python predict.py "What is photosynthesis?"
  
  # Interactive mode
  python predict.py --interactive
  
  # Predict from CSV file
  python predict.py --file data.csv --output results.csv
  
  # Predict from Excel file
  python predict.py --file data.xlsx --output results.xlsx
  
  # Specify Excel sheet
  python predict.py --file data.xlsx --sheet "Sheet2" --output results.xlsx
  
  # Different question column name
  python predict.py --file data.csv --question-column "text" --output results.csv
  
  # With confidence and probabilities
  python predict.py --confidence --probabilities "Analyze the causes of World War I"

Installation requirements:
  # For Excel support
  pip install openpyxl
        """
    )
    
    parser.add_argument(
        'text', 
        nargs='?', 
        help='Text/question to classify'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to CSV or Excel file containing questions to classify'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file containing questions to classify (deprecated, use --file)'
    )
    
    parser.add_argument(
        '--sheet',
        type=str,
        default=0,
        help='Sheet name or index for Excel files (default: 0 - first sheet)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save results (CSV or Excel format based on extension)'
    )
    
    parser.add_argument(
        '--question-column',
        type=str,
        default='question',
        help='Name of the column containing questions in CSV (default: question)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        action='store_true',
        help='Show prediction confidence'
    )
    
    parser.add_argument(
        '--probabilities', '-p',
        action='store_true',
        help='Show probabilities for all classes'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        default='best_model.pth',
        help='Path to trained model file (default: best_model.pth)'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = BloomPredictor(model_path=args.model_path)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        sys.exit(1)
    
    # Check Excel support if needed
    input_file = args.file or args.csv
    if input_file:
        file_ext = os.path.splitext(input_file)[1].lower()
        if file_ext in ['.xlsx', '.xls'] and not EXCEL_SUPPORT:
            print("Error: Excel support not available!")
            print("Please install openpyxl: pip install openpyxl")
            sys.exit(1)
    
    # File prediction mode (CSV or Excel)
    if input_file:
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found!")
            sys.exit(1)
        
        try:
            # Convert sheet argument to proper type
            sheet_arg = args.sheet
            if isinstance(sheet_arg, str) and sheet_arg.isdigit():
                sheet_arg = int(sheet_arg)
            
            results_df = predictor.predict_from_file(
                input_file, 
                args.output, 
                args.question_column,
                sheet_arg
            )
            print(f"\nPrediction completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # Interactive mode
    if args.interactive:
        predictor.interactive_mode()
        return
    
    # Single prediction mode
    if not args.text:
        print("Error: Please provide text to classify, use --file for file processing, or use --interactive mode")
        parser.print_help()
        sys.exit(1)
    
    # Make prediction
    result = predictor.predict_single(
        args.text, 
        return_confidence=args.confidence,
        return_probabilities=args.probabilities
    )
    
    # Display result
    if 'error' in result and result['error']:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print(f"\nQuestion: {result['text']}")
    print(f"Predicted Level: L{result['level']}")
    print(f"Description: {result['description']}")
    
    if 'confidence' in result:
        print(f"Confidence: {result['confidence']:.4f}")
    
    if 'probabilities' in result:
        print(f"\nProbabilities:")
        for level_key, prob in result['probabilities'].items():
            print(f"  {level_key}: {prob:.4f}")

if __name__ == "__main__":
    main()