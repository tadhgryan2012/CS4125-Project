from abc import ABC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from modelling.data_model import Data
import os
import joblib

from utils.Util import Util

class BaseModel(ABC):
    def __init__(self) -> None:
        """
        Instantiates the Model and compiles.
        """
        self.model = None

    def train(self, data: Data, save_path: str, retrain: bool = False, print_results: bool = True) -> tuple:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        If a saved model exists, it is loaded instead of training a new one.
        
        :param data: The dataset to train and test the model.
        :param save_path: Path to save or load the trained model.
        :param print_results: Whether to print evaluation metrics.
        :return: Tuple containing evaluation metrics (accuracy, precision, recall, f1, classification_report, confusion_matrix).
        """
        self.getTrainedModel(data, retrain, save_path)

        # Evaluate the model
        y_pred = self.model.predict(data.get_X_test())
        y_true = data.get_type_y_test()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        cr = classification_report(y_true, y_pred, zero_division=1)
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        if print_results: 
            print(f"Model accuracy: {accuracy:.2f}")
            print(f"Model precision: {precision:.2f}")
            print(f"Model recall: {recall:.2f}")
            print(f"Model F1-score: {f1:.2f}")
            print("\nClassification Report:")
            print(cr)
            print("\nConfusion Matrix:")
            print(cm)
        
        return accuracy, precision, recall, f1, cr, cm

    def predict(self, data: Data, save_path: str, retrain: bool = False):
        """
        Predicts using the model on the data specified.
        """
        self.getTrainedModel(data, retrain, save_path)
        if self.model is None:
            raise ValueError("The model has not been trained yet. Train the model before calling predict().")

        self.predictions = self.model.predict(data.get_X_test())
        return self.predictions

    def getTrainedModel(self, data: Data, retrain: bool, save_path: str):
        # Check if a pre-trained model already exists
        if (not retrain) and os.path.exists(save_path):
            print(f"Loading existing model from {save_path}...")
            self.model = joblib.load(save_path)
        else:
            print("Training new model...")
            # Train the model
            df_origin = Util.get_data()

            X, df = Util.get_embeddings(df_origin)
            df = df.reset_index(drop=True)
            data = Util.get_data_object(X, df)

            self.model.fit(data.get_X_train(), data.get_type_y_train())

            # Save the trained model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
        