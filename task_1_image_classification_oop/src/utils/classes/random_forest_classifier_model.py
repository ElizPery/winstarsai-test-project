from .mnist_classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierModel(MnistClassifierInterface):
    """Implementation of classification using a Random Forest."""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        """Trains the Random Forest model."""
        # As Random Forest requires flattened data use reshape function
        X_flat = X_train.reshape(X_train.shape[0], -1)
        print("Training Random Forest...")
        self.model.fit(X_flat, y_train)
        print("Random Forest training complete.")

    def predict(self, X_test):
        """Predicts labels using the trained Random Forest model."""
        # Flatten test data
        X_flat = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_flat)