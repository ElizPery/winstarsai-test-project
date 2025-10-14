import argparse
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import os
import sys

# Add necessary paths to import inference modules
sys.path.append(os.path.dirname('src'))

from src.constants.constants import MODEL_NAME

# Configuration
LABEL_NAMES = ['O', 'B-ANIMAL', 'I-ANIMAL']
ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}

def train_ner_model(args):
    """
    Main function to load, tokenize, and train the NER model.
    """
    print("--- Starting NER Model Training ---")
    
    # 1. Load Data
    try:
        data_path = os.path.join(os.path.dirname(__file__).replace('src/models/ner_model', 'src/data/ner_data'), 'data_ner.csv')
        df = pd.read_csv(data_path)
        # Use literal_eval for safer conversion if needed, but simple eval works for this structure
        df['tokens'] = df['tokens'].apply(eval) 
        df['tags'] = df['tags'].apply(eval)
    except FileNotFoundError:
        print("Error: 'ner_data.csv' not found. Please run ner_data_prep.py first.")
        return

    # 2. Prepare Hugging Face Dataset
    # Define features, noting that ClassLabel will convert string tags to integer IDs upon creation.
    features = Features({
        'id': Value('int32'),
        'tokens': Sequence(Value('string')),
        'tags': Sequence(ClassLabel(names=LABEL_NAMES))
    })
    
    data_dict = {'id': list(range(len(df))), 'tokens': df['tokens'].tolist(), 'tags': df['tags'].tolist()}
    dataset = Dataset.from_dict(data_dict, features=features)
    
    # 3. Tokenizer and Data Alignment
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align_labels(examples):
        """Tokenize and align the word-level tags (now integer IDs) to the sub-word tokens."""
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["tags"]):
            # 'label' here is a list of INTEGER IDs (e.g., [0, 1, 0, 0])
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Ignore special tokens
                elif word_idx != previous_word_idx:
                    # FIX: Use the integer ID directly, as 'label[word_idx]' is already an integer.
                    # Previous code: label_ids.append(LABEL2ID[label[word_idx]]) 
                    label_ids.append(label[word_idx]) 
                else:
                    label_ids.append(-100) # Only first subtoken gets the label
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply the tokenization and alignment
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    
    # Split into mock train/test
    train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # 4. Model Setup
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_NAMES), 
        id2label=ID2LABEL, 
        label2id=LABEL2ID
    )

    # 5. Initialize Data Collator
    # This collator correctly handles padding of input_ids, attention_mask, and labels (-100).
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 6. Training Arguments and Trainer
    training_args = TrainingArguments(
        output_dir='./ner_results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./ner_logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7. Train Model
    print("\nTraining BERT Model...")
    trainer.train()

    # 8. Save Model
    os.makedirs(args.model_path, exist_ok=True)
    model.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    print(f"\nNER model and tokenizer saved to: {args.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a transformer-based NER model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--model_path", type=str, default="./model_artifacts/ner_model", help="Path to save the trained model.")
    args = parser.parse_args()
    
    train_ner_model(args)