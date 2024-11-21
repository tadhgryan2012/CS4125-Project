import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)

# This file already contain the code for implementing randomforest model
# Carefully observe the methods below and try calling them in modelling.py

class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test: pd.Series):
    
        if self.mdl is None:
            raise ValueError("The model has not been trained yet. Train the model before calling predict().")

        print(f"Making predictions with RandomForest model on data of shape: {X_test.shape}")
        self.predictions = self.mdl.predict(X_test)  # Generate predictions
        print(f"Predictions generated: {self.predictions}")
        return self.predictions  #returns predictions

    def print_results(self, data):
   
        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")
    
        print("RandomForest Evaluation:")
        print(classification_report(data.get_type_y_test(), self.predictions))


    def data_transform(self) -> None:
        ...

