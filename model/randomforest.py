import numpy as np
from model.base import BaseModel
from modelling.data_model import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random
seed = 0
np.random.seed(seed)
random.seed(seed)

# This file already contain the code for implementing randomforest model
# Carefully observe the methods below and try calling them in modelling.py

class RandomForest(BaseModel):
    def __init__(self) -> None:
        super(RandomForest, self).__init__()
        self.model = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')

    def train(self, data: Data) -> None:
        self.model.fit(data.get_X_train(), data.get_type_y_train())
    
        # Get predictions on the test set
        y_pred = self.model.predict(data.get_X_test())
        print(f"Model predictions: {y_pred}")   
         
        # Get the actual test labels
        y_true = data.get_type_y_test()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        
        print(f"Model accuracy: {accuracy:.2f}")
        print(f"Model precision: {precision:.2f}")
        print(f"Model recall: {recall:.2f}")
        print(f"Model F1-score: {f1:.2f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=1))

    def predict(self, data: Data):
        if self.model is None:
            raise ValueError("The model has not been trained yet. Train the model before calling predict().")

        self.predictions = self.model.predict(data.get_X_test())  # Generate predictions
        return self.predictions  #returns predictions
