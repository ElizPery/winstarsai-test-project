from task_1_image_classification_oop.src.utils.classes.random_forest_classifier_model import RandomForestClassifierModel
from src.utils.classes.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.utils.classes.cnn_classifier import CNNClassifier

ALGORITHMS = {
        'rf': RandomForestClassifierModel,
        'nn': FeedForwardNNClassifier,
        'cnn': CNNClassifier
    }