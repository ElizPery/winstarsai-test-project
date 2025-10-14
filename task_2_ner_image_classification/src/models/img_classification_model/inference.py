import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- Configuration Constants ---
MODEL_ARTIFACTS_DIR = './model_artifacts/img_model'
IMG_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'ic_model_weights.pth')
IMG_MAPPING_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'ic_label_mapping.txt')
IMG_RAW_DIR = os.path.dirname(__file__).replace('src/models/img_classification_model', 'src/data/img_data_raw') # Base directory needed for file system checks

# Global variables for model caching
GLOBAL_IC_MODEL = None
GLOBAL_CLASS_NAMES = []
GLOBAL_TRANSFORM = None

def load_ic_model():
    """
    Loads the trained Image Classification model and class mapping.
    Uses global caching to ensure the model is loaded only once.
    """
    global GLOBAL_IC_MODEL, GLOBAL_CLASS_NAMES, GLOBAL_TRANSFORM

    if GLOBAL_IC_MODEL is not None:
        return GLOBAL_IC_MODEL, GLOBAL_CLASS_NAMES, GLOBAL_TRANSFORM
    
    print("--- Loading Image Classification Model ---")

    # 1. Load Class Names from Text File (Robust Mapping)
    if not os.path.exists(IMG_MAPPING_PATH):
        print(f"Error: Label mapping file not found at {IMG_MAPPING_PATH}. Please train the IC model first.")
        return None, None, None
        
    try:
        with open(IMG_MAPPING_PATH, 'r') as f:
            # Read and strip newline characters
            CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
        NUM_CLASSES = len(CLASS_NAMES)
        GLOBAL_CLASS_NAMES = CLASS_NAMES
        print(f"Loaded {NUM_CLASSES} classes from mapping file: {CLASS_NAMES}")
    except Exception as e:
        print(f"Error reading class mapping: {e}")
        return None, None, None

    # 2. Define Model Architecture (MUST match train.py)
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None) # Load structure without default weights
        
        # Replace the final fully connected layer using the dynamically determined NUM_CLASSES
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        model = model.to(device)

        # Load trained weights
        if not os.path.exists(IMG_MODEL_PATH):
             print(f"Error: Model weights not found at {IMG_MODEL_PATH}. Please train the IC model first.")
             return None, None, None

        model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location=device))
        model.eval()
        
        GLOBAL_IC_MODEL = model
        print("Image Classification model loaded successfully.")

    except Exception as e:
        print(f"Error loading IC model or weights: {e}")
        return None, None, None

    # 3. Define Transformations (MUST match train.py validation transforms)
    GLOBAL_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return GLOBAL_IC_MODEL, GLOBAL_CLASS_NAMES, GLOBAL_TRANSFORM

def infer_class_from_image(image_path):
    """
    Performs image classification inference on a given image file path.

    Args: image_path (str): The full or relative path to the image file.

    Returns: tuple[str, float]: The predicted animal class name and its confidence score.
    """
    # Load model and components (uses caching)
    model, class_names, data_transform = load_ic_model()
    
    if model is None:
        return "Unknown_Animal", 0.0 # Return default if model failed to load

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load and Preprocess Image
    try:
        # Check if path is relative, adjust if necessary (though usually the pipeline passes the full path)
        if not os.path.exists(image_path):
            # Attempt to resolve path relative to RAW_IMAGES_DIR, useful for local testing
            full_path = os.path.join(IMG_RAW_DIR, image_path)
            if os.path.exists(full_path):
                image_path = full_path
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transform(image).unsqueeze(0).to(device)

    except FileNotFoundError as e:
        print(f"IC Inference Error: {e}. Cannot open image.")
        return "Unknown_Animal", 0.0
    except Exception as e:
        print(f"IC Inference Error during image loading/preprocessing: {e}")
        return "Unknown_Animal", 0.0


    # 2. Predict
    with torch.no_grad():
        output = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get the highest probability and index
        confidence, predicted_index = torch.max(probabilities, 1)

    # 3. Map Index to Class Name
    predicted_class = class_names[predicted_index.item()]
    predicted_confidence = confidence.item()

    return predicted_class, predicted_confidence

if __name__ == '__main__':
    # --- Example Usage (REQUIRES TRAINED MODEL AND MAPPING FILE) ---
    print("This is the Image Classification Inference test script.")
    print("It requires a trained model ('ic_model_weights.pth') and mapping file ('ic_label_mapping.txt').")
    
    # Mock file path
    mock_image_path = os.path.join(os.path.dirname(__file__).replace('src/models/img_classification_model', 'src/data/img_data_raw/Tiger'), 'Tiger_0220.jpg')
    
    # Create a dummy structure for testing if the model files exist
    if not os.path.exists(IMG_MODEL_PATH) or not os.path.exists(IMG_MAPPING_PATH):
         print("\nSkipping live test: Model or Mapping file not found. Run training first.")
    else:
        # Assuming image not found at this path:
        if not os.path.exists(mock_image_path):
            print(f"\nSkipping live test: Mock image not found at {mock_image_path}. Please replace with a real path.")
        else:
            # Loads the model and predicts the class
            predicted_class, confidence = infer_class_from_image(mock_image_path)
            print(f"\nTest: Predicted: {predicted_class}, Confidence: {confidence:.4f}")
            