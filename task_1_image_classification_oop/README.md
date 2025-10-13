# Task 1: Image Classification + OOP

This project implements an image classification solution for the **MNIST dataset** using three different algorithms: **Random Forest (RF)**, a **Feed-Forward Neural Network (NN)**, and a **Convolutional Neural Network (CNN)**. The solution adheres to strict Object-Oriented Programming principles by defining a common interface and a facade class.

## **Design Overview**

The core of the design revolves around a few key components:

1.  **`MnistClassifierInterface`**: An abstract base class that defines the contract (`train` and `predict` methods) that all concrete classifier implementations must follow.
2.  **Concrete Classifiers (`RandomForestClassifier`, `FeedForwardNNClassifier`, `CNNClassifier`)**: These classes implement the `MnistClassifierInterface` and contain the logic specific to each model, including data preprocessing tailored for that model (e.g., flattening for RF/NN, retaining 2D structure for CNN).
3.  **`MnistClassifier` (Facade)**: This class acts as a central access point. It takes the algorithm name (`rf`, `nn`, or `cnn`) as an input and delegates the `train` and `predict` calls to the correct underlying concrete classifier object. This hides the implementation details from the user, fulfilling the requirement for a consistent input/output structure regardless of the chosen algorithm.

## **Setup and Installation**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ElizPery/winstarsai-test-project.git
    cd task_1_image_classification_oop
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

4.  **Run the Demo:**
    Launch the Jupyter Notebook to see the solution in action, including training, prediction examples, and edge case demonstrations.
    ```bash
    jupyter notebook demo_notebook.ipynb
    ```