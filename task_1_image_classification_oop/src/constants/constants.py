from src.utils.classes.random_forest_classifier import RandomForestClassifier
from src.utils.classes.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.utils.classes.cnn_classifier import CNNClassifier

ALGORITHMS = {
        'rf': RandomForestClassifier,
        'nn': FeedForwardNNClassifier,
        'cnn': CNNClassifier
    }