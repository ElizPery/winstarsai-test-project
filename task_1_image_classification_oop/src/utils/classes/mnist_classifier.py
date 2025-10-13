from src.constants.constants import ALGORITHMS

class MnistClassifier:
    """
    Class that hides 3 different classification algorithms such as Random Forest, Feedforward Neural Network, and Convolutional Neural Network.
    It provides a uniform interface (train/predict) regardless of the algorithm chosen.
    """
    def __init__(self, algorithm: str, **kwargs):
        """
        Initializes the MnistClassifier with the chosen algorithm.
        Args:
            algorithm (str): The classification algorithm to use ('rf', 'nn', or 'cnn').
            **kwargs: Arguments passed to the underlying model's constructor.
        """
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be one of {list(ALGORITHMS.keys())}")
        
        self.algorithm_name = algorithm
        self.classifier_instance = ALGORITHMS[algorithm](**kwargs)

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the selected model.
        Args: X_train - training features (images).
              y_train - training labels.
            **kwargs: Additional training parameters (e.g., epochs, batch_size).
        """
        self.classifier_instance.train(X_train, y_train, **kwargs)

    def predict(self, X_test):
        """
        Makes predictions using the selected model.
        Args: X_test - test features (images).
        Returns: predicted class labels.
        """
        return self.classifier_instance.predict(X_test)