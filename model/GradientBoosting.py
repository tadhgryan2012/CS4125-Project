from sklearn.ensemble import GradientBoostingClassifier
from model.base import BaseModel
from modelling.data_model import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class GradientBoosting(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3) -> None:
        super(GradientBoosting, self).__init__()
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.predictions = None

    def train(self, data: Data) -> None:
        print("Training Gradient Boosting model...")
        self.model.fit(data.get_X_train(), data.get_type_y_train())
        print("Gradient Boosting training complete.")

        # Generate predictions for the test set
        y_pred = self.model.predict(data.get_X_test())
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

        print("Making predictions with Gradient Boosting model...")
        self.predictions = self.model.predict(data.get_X_test())
        print(f"Predictions generated: {self.predictions}")
        return self.predictions
