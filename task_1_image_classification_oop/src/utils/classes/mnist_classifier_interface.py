from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Interface (Abstract Base Class) for all MNIST classification models.
    Defines the contract for training and prediction.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the classification model.
        Args: X_train - training features.
              y_train - training labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Makes predictions on new data.
        Args: X_test - test features. 
        Returns: predicted class labels.
        """
        pass