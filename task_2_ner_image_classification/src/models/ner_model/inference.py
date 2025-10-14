import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

# Configuration (must match train.py)
LABEL_NAMES = ['O', 'B-ANIMAL', 'I-ANIMAL']
ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}

# Global variables to store the loaded model and tokenizer
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

def load_ner_model(model_path=os.path.dirname(__file__).replace('src/models/ner_model', 'model_artifacts/ner_model')):
    """Loads the trained transformer model and tokenizer."""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER
    
    if GLOBAL_MODEL is not None:
        return GLOBAL_MODEL, GLOBAL_TOKENIZER # Return the loaded objects

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the saved model and tokenizer from the specified path
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.eval()
        GLOBAL_MODEL = model
        GLOBAL_TOKENIZER = tokenizer
        print(f"NER Transformer model loaded successfully from {model_path}.")
        return model, tokenizer
        
    except Exception as e:
        # Handle case where the model hasn't been trained/saved yet
        print(f"Warning: NER model not found at {model_path}. Returning mock functionality. Error: {e}")
        return None, None # Indicate failure to load

def infer_animal_from_text(text_message: str) -> tuple[str, float]:
    """
    Predicts the animal entity from the text using the loaded transformer model.
    Returns the extracted animal name and its confidence (using the max logit).
    """
    model, tokenizer = load_ner_model()
    
    # --- Fallback for missing model ---
    if model is None:
        mock_entities = {
            "cow": "Cow", "dog": "Dog", "cat": "Cat", "horse": "Horse",
            "sheep": "Sheep", "bird": "Bird", "squirrel": "Squirrel",
            "elephant": "Elephant", "fox": "Fox", "deer": "Deer",
            "black bear": "Black Bear", "red panda": "Red Panda", 
            "mountain goat": "Mountain Goat", "gray wolf": "Gray Wolf"
        }
        # Simple keyword matching for fallback
        for entity, formatted_name in mock_entities.items():
            if entity in text_message.lower():
                return formatted_name, 0.75 # Return mock confidence
        return "", 0.0

    device = next(model.parameters()).device

    # Tokenize input text, splitting into words first (like training)
    words = text_message.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    
    # *** FIX APPLIED HERE ***
    # 1. Get word_ids from the BatchEncoding object BEFORE converting to dict/tensor
    #    We must ensure word_ids corresponds to the first (and only) item in the batch (index 0).
    word_ids = inputs.word_ids(batch_index=0) 
    
    # 2. Convert to dict and move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    
    # Map predictions back to original words and extract entity
    
    extracted_entity_tokens = []
    
    for token_idx in range(len(word_ids)):
        token_word_idx = word_ids[token_idx]
        
        if token_word_idx is None:
            continue
        
        # Only check the prediction for the first sub-token of each word
        # Ensure we are checking the beginning of a word block or the first token overall
        is_first_subtoken = (token_idx == 0) or (word_ids[token_idx] != word_ids[token_idx - 1])
        
        if is_first_subtoken:
            tag_id = predictions[token_idx]
            tag = ID2LABEL[tag_id]
            
            # Get the actual word corresponding to this word_id
            word = words[token_word_idx]
            
            if tag.startswith('B-ANIMAL'):
                extracted_entity_tokens = [word]
            elif tag.startswith('I-ANIMAL') and extracted_entity_tokens:
                extracted_entity_tokens.append(word)
            elif tag == 'O' and extracted_entity_tokens and len(extracted_entity_tokens) > 0:
                # Stop if O is encountered after B/I tags
                break
    
    extracted_entity = " ".join(extracted_entity_tokens)
    
    # Confidence calculation: Use the max probability of the extracted entity tokens
    confidence = 0.0
    if extracted_entity:
        # Get probabilities for the predicted sequence
        softmax_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # We'll use a simple proxy: the average confidence for the tokens tagged B-ANIMAL/I-ANIMAL
        relevant_probs = []
        for token_idx in range(len(word_ids)):
            token_word_idx = word_ids[token_idx]
            if token_word_idx is None:
                continue
            
            tag_id = predictions[token_idx]
            if ID2LABEL[tag_id].endswith('-ANIMAL'):
                # Confidence is the probability of the predicted tag
                relevant_probs.append(softmax_probs[token_idx, tag_id])

        if relevant_probs:
            confidence = float(sum(relevant_probs) / len(relevant_probs))
        
    return extracted_entity, confidence

if __name__ == '__main__':
    # Example tests (will use mock if training hasn't been run)
    print(f"Test 1: {infer_animal_from_text('I see a big Red Panda here.')}")
    print(f"Test 2: {infer_animal_from_text('Is that a dog or a cat?')}")
    print(f"Test 3: {infer_animal_from_text('There is a large Gray Wolf in the forest.')}")