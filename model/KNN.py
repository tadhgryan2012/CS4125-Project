from sklearn.neighbors import KNeighborsClassifier
from model.base import BaseModel
<<<<<<< HEAD
from modelling.data_model import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

=======
>>>>>>> main

class KNN(BaseModel):
    def __init__(self, n_neighbors=5, metric='minkowski', p=2) -> None:
        super(KNN, self).__init__()
        self.model = KNeighborsClassifier(
<<<<<<< HEAD
            n_neighbors=n_neighbors,  # Number of neighbors
            metric=metric,  # Distance metric ('minkowski' is default)
            p=p  # Power parameter for Minkowski metric (p=2 is Euclidean distance)
        )
        self.predictions = None

    def train(self, data: Data) -> None:
        self.model.fit(data.get_X_train(), data.get_type_y_train())

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

        print("Making predictions with K-Nearest Neighbors model...")
        self.predictions = self.model.predict(data.get_X_test())
        print(f"Predictions generated: {self.predictions}")
        return self.predictions
=======
            n_neighbors=n_neighbors,
            metric=metric,
            p=p
        )
        self.predictions = None
>>>>>>> main
