from model.RandomForest import RandomForest
from model.LogisticRegression import LogisticRegression
from model.GradientBoosting import GradientBoosting
from model.SVM import SVM
from model.KNN import KNN
from model.NaiveBayes import NaiveBayes

from modelling.data_model import Data
import numpy as np

class ModelFactory:
    """A Singleton and Factory design pattern for classification models."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance of the ModelFactory exists (Singleton)."""
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Maps bitmask to corresponding model
        self.model_map = {
            0b001: RandomForest,
            0b010: LogisticRegression,
            0b011: SVM,
            0b100: GradientBoosting,
            0b101: KNN,
            0b110: NaiveBayes,

        }
        self.model = None

    def create_model(self, bitmask):
        """
        Create a model based on the bitmask.

        Args:
            bitmask (int): A binary bitmask to select the model.

        Returns:
            model: An untrained model instance.
        """
        if bitmask not in self.model_map:
            raise ValueError(f"Invalid bitmask {bin(bitmask)}. Available models: {list(self.model_map.keys())}")
        self.model = self.model_map[bitmask]()
        print(f"Created model: {self.model.__class__.__name__}")

    def train_evaluate(self, data: Data):
        """
        Compiles, trains, and evaluates the model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            test_size (float): Proportion of data to use as test set.
            random_state (int): Random seed for reproducibility.

        Returns:
            float: Accuracy score on the test data.
        """
        if self.model is None:
            raise ValueError("No model created. Use 'create_model' first.")

        # Train and evaluate
        self.model.train(data)

    def predict(self, data: Data):
        """
        Makes predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix for predictions.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self.model is None:
            raise ValueError("No model created. Use 'create_model' first.")
        predictions = self.model.predict(data)
        print(f"Predictions: {predictions}")
        return predictions
