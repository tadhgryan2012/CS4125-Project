from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model.randomforest import RandomForest


def model_predict(data, df, name):
    print("Starting model_predict...")
    # Init Rainforest & train it.
    model = RandomForest(
        model_name='RandomForest',
        embeddings=data.get_embeddings(),
        y=data.get_type()
    )
    model.train(data)  
    
    # Predict on test data
    predictions = model.predict(data.get_X_test())  
    print(f"Predictions generated in model_predict: {predictions}")
    return predictions, model

def model_evaluate(model, data):
    print("Starting model_evaluate...")
    
    # Get predictions on the test set
    y_pred = model.predict(data.get_X_test())


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
    
    # Optional: Print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=1))
