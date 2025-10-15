import os

# Import inference functions from the respective modules
from models.ner_model.inference import infer_animal_from_text
from models.img_classification_model.inference import infer_class_from_image

def run_pipeline(text_message: str, image_path: str, confidence_threshold: float = 0.5) -> bool:
    """
    Main pipeline function to check if the animal mentioned in the text 
    matches the animal in the image.
    """
    
    # 1. Named Entity Recognition (NER)
    # Predicted animal and confidence (e.g., 'rabbit', 0.95)
    try:
        text_animal_entity, ner_confidence = infer_animal_from_text(text_message)
    except Exception as e:
        print(f"NER Error: {e}")
        return False # Cannot proceed without a recognized animal

    # Check if a valid animal entity was extracted
    if not text_animal_entity:
        print("Pipeline failed: No animal entity found in the text message.")
        return False
        
    # 2. Image Classification (IC)
    # Predicted animal and confidence (e.g., 'rabbit', 0.95)
    try:
        predicted_animal, ic_confidence = infer_class_from_image(image_path)
    except Exception as e:
        print(f"IC Error: {e}")
        return False

    # 3. Comparison and Verification Logic
    
    # Standardize the names for comparison (e.g., lowercasing)
    # Example: 'Red Panda' -> 'red panda'
    text_animal = text_animal_entity.lower()
    image_animal = predicted_animal.lower()

    # Apply a confidence check 
    if ner_confidence < confidence_threshold:
        print(f"Warning: Low confidence in NER prediction ({ner_confidence:.2f}).")
    if ic_confidence < confidence_threshold:
        print(f"Warning: Low confidence in IC prediction ({ic_confidence:.2f}).")
        
    # Final Decision
    # Comparing the normalizing names and ensuring both models are sufficiently confident
    is_correct = (text_animal == image_animal) and \
                 (ner_confidence >= confidence_threshold) and \
                 (ic_confidence >= confidence_threshold)

    print(f"\n--- Pipeline Output ---")
    print(f"Text Entity: '{text_animal_entity}' (Conf: {ner_confidence:.2f})")
    print(f"Image Class: '{predicted_animal}' (Conf: {ic_confidence:.2f})")
    print(f"Confidence Threshold Met: {ner_confidence >= confidence_threshold and ic_confidence >= confidence_threshold}")
    print(f"Match Found: {is_correct}")
    
    return is_correct

if __name__ == '__main__':
    print("Initializing pipeline for testing...")

    # Demonstration image paths
    mock_rabbit_path = os.path.join(os.path.dirname(__file__), 'data/img_data_raw/Rabbit/Rabbit_0016.jpg')
    mock_duck_path = os.path.join(os.path.dirname(__file__), 'data/img_data_raw/Duck/Duck_0322.jpg')

    # Case 1: Perfect Match (Assuming IC predicts 'Rabbit' for mock_rabbit_path)
    print("\n--- Running Test Case 1: Perfect Match (Rabbit) ---")
    result_match = run_pipeline(
        text_message="I definitely see a rabbit in the pasture, it is large.",
        image_path=mock_rabbit_path 
    )
    print(f"\nResult (Match): {result_match}")
    
    # Case 2: Mismatch (Assuming IC predicts 'Duck' for mock_duck_path)
    print("\n--- Running Test Case 2: Mismatch (Horse vs duck) ---")
    result_mismatch = run_pipeline(
        text_message="That must be a horse over there.",
        image_path=mock_duck_path 
    )
    print(f"\nResult (Mismatch): {result_mismatch}")