import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_classification_data(
        raw_data_dir = os.path.dirname(__file__).replace('src/models/img_classification_model', 'src/data/img_data_raw'), 
        output_path = os.path.dirname(__file__).replace('src/models/img_classification_model', 'src/data/img_data_csv')
        ):
    """
    Prepares data for Image Classification by extracting labels from the parent directory names.

    Assumes data is organized as: img_data_raw/CLASSNAME/image.jpg
    The CLASSNAME folder name is used as the label.
    """
    print("--- Starting Image Classification Data Preparation (Directory Based Parsing) ---")
    
    # Check for Directory Existence
    if not os.path.isdir(raw_data_dir):
         print(f"Error: Raw data directory not found at '{raw_data_dir}'. Please ensure your image classes are structured inside this folder.")
         return

    # Walk the Directory and Extract Labels
    data = []
    VALID_EXTENSIONS = ('.jpg', '.png', '.jpeg')
    
    # os.walk traverses the directory tree
    for root, _, files in os.walk(raw_data_dir):
        # The class label is the name of the immediate directory containing the files
        class_label = os.path.basename(root)
        
        # Skip the root directory itself if it contains files (it should only contain subdirectories)
        if root == raw_data_dir:
            continue

        for filename in files:
            if filename.lower().endswith(VALID_EXTENSIONS):
                # The 'filename' column stores the full relative path for the model to load later
                # We store 'CLASSNAME/image.jpg' relative to the base 'data/img_data_raw' folder
                relative_path = os.path.join(class_label, filename)
                data.append({'filename': relative_path, 'class_label': class_label})

    df = pd.DataFrame(data)
    
    if df.empty:
        print(f"Error: DataFrame is empty. Check that '{raw_data_dir}' contains subdirectories with images inside.")
        return

    print(f"Found {len(df)} images belonging to {df['class_label'].nunique()} unique classes.")
    print(f"Detected classes: {', '.join(df['class_label'].unique())}")


    # Split Data and Save CSVs
    os.makedirs(output_path, exist_ok=True)
    
    # Stratified split to ensure equal representation of classes
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42, 
            stratify=df['class_label']
        )
    except ValueError as e:
        print(f"Warning: Could not perform stratified split (too few samples per class?). Falling back to non-stratified split. Error: {e}")
        train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
        
    train_df.to_csv(os.path.join(output_path, 'img_data_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'img_data_test.csv'), index=False)

    print(f"Train data map saved: {len(train_df)} samples.")
    print(f"Test data map saved: {len(test_df)} samples.")
    print("Image data preparation complete.")

if __name__ == '__main__':
    # Ensure your images are structured like: ./src/data/img_data_raw/Bear/001.jpg
    prepare_classification_data() 