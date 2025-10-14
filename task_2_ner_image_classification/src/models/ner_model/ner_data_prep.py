import pandas as pd
import numpy as np
import os
import sys

# Add necessary paths to import inference modules
sys.path.append(os.path.dirname('src'))
sys.path.append(os.path.dirname('data_ner'))

from src.constants.constants import NER_ANIMALS, NER_TEMPLATES

def generate_ner_data(num_samples_per_class=20):
    """
    Generates mock NER data (sentences and BIO tags) for the animal classes,
    including robust tagging for multi-word entities.
    """
    print(f"Generating mock NER data for {len(NER_ANIMALS)} classes...")
    
    data = []

    for animal in NER_ANIMALS:
        for _ in range(num_samples_per_class):
            template = np.random.choice(NER_TEMPLATES)
            
            # Mix casing for better robustness
            animal_name = animal if np.random.rand() < 0.8 else animal.lower() 
            
            sentence = template.format(animal_name)
            
            # Tokenize the sentence and generate initial 'O' tags
            tokens = sentence.split()
            tags = ['O'] * len(tokens)
            
            # Split the animal name into component words
            animal_words = animal_name.split()
            
            # Find the starting index of the first word of the animal entity
            try:
                # Search for the first word in the sentence tokens, ignoring punctuation and case
                start_index = next(
                    i for i, token in enumerate(tokens) 
                    if token.lower().strip('.,!?') == animal_words[0].lower().strip('.,!?')
                )
            except StopIteration:
                # Skip if the entity's first word isn't found in the tokenized sentence
                continue

            # Apply B-ANIMAL tag to the first word
            tags[start_index] = 'B-ANIMAL'
            
            # Apply I-ANIMAL tags to subsequent words (if multi-word)
            if len(animal_words) > 1:
                for i in range(1, len(animal_words)):
                    current_index = start_index + i
                    # Check bounds and verify the token matches the expected word
                    if (current_index < len(tokens) and 
                        tokens[current_index].lower().strip('.,!?') == animal_words[i].lower().strip('.,!?')):
                        
                        tags[current_index] = 'I-ANIMAL'
                    else:
                        # If the sequence breaks (e.g., unexpected punctuation), stop tagging this entity
                        break
            
            data.append({'tokens': tokens, 'tags': tags})

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save the data
    output_path = os.path.join(os.path.dirname(__file__).replace('src/models/ner_model', 'src/data/ner_data'), 'data_ner.csv')
    df.to_csv(output_path, index=False)
    print(f"Mock NER data saved to: {output_path} ({len(df)} samples)")
    return df

if __name__ == '__main__':
    generate_ner_data()
