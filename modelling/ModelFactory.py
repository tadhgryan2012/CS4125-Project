from model.RandomForest import RandomForest
from model.LogisticRegression import LogisticRegressionModel
from model.GradientBoosting import GradientBoosting
from model.SVM import SVM
from model.KNN import KNN
from model.NaiveBayes import NaiveBayes

from modelling.data_model import Data

class ModelFactory:
    """A Singleton and Factory design pattern for classification models."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance of the ModelFactory exists (Singleton)."""
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_map = {
<<<<<<< HEAD
            0: RandomForest,
            1: LogisticRegressionModel,
            2: SVM,
            3: GradientBoosting,
            4: KNN,
            5: NaiveBayes,
=======
            0: (RandomForest, "data/models/random_forest_model.pkl"),
            1: (LogisticRegressionModel, "data/models/logistic_regression_model.pkl"),
            2: (SVM, "data/models/svm_model.pkl"),
            3: (GradientBoosting, "data/models/gradient_boosting_model.pkl"),
            4: (KNN, "data/models/knn_model.pkl"),
            5: (NaiveBayes, "data/models/naive_bayes_model.pkl"),
>>>>>>> main
        }
        self.models = []

    def create_model(self, bitmask):
        """
        Create a model based on the bitmask.

        Args:
            bitmask (int): A binary bitmask to where each bit corresponds to a model.

        Returns:
            model list: Of the models selected in the bitmask
        """
        self.models = []
<<<<<<< HEAD
        for bit_position, model_class in self.model_map.items():
            if bitmask & (1 << bit_position):
                model_instance = model_class()
                self.models.append(model_instance)
                print(f"Created model: {model_instance.__class__.__name__}")

    def train_evaluate(self, data: Data):
=======
        for bit_position, (model_class, save_path) in self.model_map.items():
            if bitmask & (1 << bit_position):
                model_instance = model_class()
                self.models.append((model_instance, save_path))
                print(f"Created model: {model_instance.__class__.__name__}")

    def train_evaluate(self, data: Data, retrain: bool = False, print_results: bool = True):
>>>>>>> main
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
        if self.models is None:
            raise ValueError("No model created. Use 'create_model' first.")

        # Train and evaluate
<<<<<<< HEAD
        for model in self.models:
            print(f"Training and evaluating model: {model.__class__.__name__}")
            model.train(data)
=======
        for model, save_path in self.models:
            print(f"Processing model: {model.__class__.__name__}")
            model.train(data, save_path=save_path, retrain=retrain,  print_results=print_results)
>>>>>>> main

    def predict(self, data: Data):
        """
        Makes predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix for predictions.

        Returns:
            np.ndarray: Predicted labels.
        """

        predictions = {}
        for model in self.models:
<<<<<<< HEAD
            predictions[model.__class__.__name__] = model.predict(data)
            print(f"Predictions for {model.__class__.__name__}: {predictions[model.__class__.__name__]}")
=======
            predictions[model[0].__class__.__name__] = model[0].predict(data)
>>>>>>> main
        return predictions
