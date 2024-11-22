from abc import ABC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from modelling.data_model import Data

class BaseModel(ABC):
    def __init__(self) -> None:
        """
        Instantiates the Model and compiles.
        """
        pass

    def train(self, data: Data, print_results: bool = True) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        self.model.fit(data.get_X_train(), data.get_type_y_train())

        y_pred = self.model.predict(data.get_X_test())
         
        # Get the actual test labels
        y_true = data.get_type_y_test()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        cr = classification_report(y_true, y_pred, zero_division=1)
        cm = confusion_matrix(y_true, y_pred)
        
        if (print_results): 
            print(f"Model accuracy: {accuracy:.2f}")
            print(f"Model precision: {precision:.2f}")
            print(f"Model recall: {recall:.2f}")
            print(f"Model F1-score: {f1:.2f}")
        
            print("\nClassification Report:")
            print(cr)
            
            print("\nConfusion Matrix:")
            print(cm)
        return accuracy, precision, recall, f1, cr, cm

    def predict(self, data: Data):
        """
        Predicts using the model on the data specified.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Train the model before calling predict().")

        self.predictions = self.model.predict(data.get_X_test())
        return self.predictions

