
# Task 2: Named Entity Recognition + Image Classification Pipeline

This project implements a multi-modal pipeline to identify animal species using a combination of **Text Analysis (Named Entity Recognition)** and **Image Classification (Computer Vision)**. The goal is to verify a statement about an animal in an image.

## **Project Flow**

1.  **NER Model**: Extracts the primary animal name (e.g., 'bear') from the input text ("There is a bear in the picture.").
2.  **Image Classification Model**: Predicts the animal class in the input image (e.g., 'bear').
3.  **Pipeline**: Compares the extracted entity from NER with the predicted class from IC, returning a boolean (True/False).

## **Dataset (Assumed Classes)**

This solution assumes the Kaggle dataset contains the following 10 animal classes, which are used throughout the mock data and model definitions:
['Bear', 'Deer', 'Duck', 'Fox', 'Parrot', 'Rabbit', 'Raccoon', 'Red panda', 'Squirrel', 'Tiger']. Link of the dataset: https://www.kaggle.com/datasets/giotamoraiti/animal-object-detection-dataset-10-classes/data

## **Setup and Installation**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ElizPery/winstarsai-test-project.git
    cd task_2_ner_image_classification
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Preparation**

    **A. Image Data Preparation:**

    The following script simulates converting dataset structure into a classification-friendly CSV. Please ensure that raw images dataset is in the `./src/data/img_data_raw directory`.
    ```bash
    python ./src/models/img_classification_model/img_data_prep.py
    ```

    **B. NER Data Preparation:**

    The NER model requires labeled text data. This script generates sample sentences using the 10 animal classes and applies BIO tagging.
    ```bash
    python ./src/models/ner_model/ner_data_prep.py
    ```

5.  **Model Training**

    **A. Train Image Classification Model (ResNet-18)**

    This script trains a ResNet-18 model on the prepared image data. Training parameters are configurable via arguments.
    ```bash
    python ./src/models/img_classification_model/train.py --epochs 3 --batch_size 16 --lr 1e-3 --workers 4
    ```

    **B. Train Named Entity Recognition Model (BERT)**

    This script fine-tunes a pre-trained BERT model for token classification on the generated NER data.
    ```bash
    python ./src/models/ner_model/train.py --epochs 3 --batch_size 16 --lr 1e-3 ----model_path ./model_artifacts/ner_model
    ```
6.  **Running the Pipeline:**
    The `./src/main.py` integrates the inference from both models. It takes the text and image path as input and outputs a boolean value.
    ```bash
    python ./src/main.py
    ```

7.  **Run the EDA and Demo:**
    All exploratory data analysis and demonstrations of the final pipeline, including edge cases, are contained within the Jupyter Notebook.
    ```bash
    jupyter notebook demo_notebook.ipynb
    ```