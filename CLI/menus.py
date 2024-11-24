import os
import pickle
from modelling.data_model import *
from modelling.ModelFactory import *
from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess.preprocess import preprocess_data
from preprocess.strategies import DeduplicationStrategy, NoiseRemovalStrategy, TranslationStrategy

def main_menu():
    while True:
        print("\nMain Menu")
        print("1) Preprocess the Data")
        print("2) Train a model")
        print("3) Classify emails")
        print("4) Exit")
        choice = int(input("Enter your choice : "))
        match choice:
            case 1: 
                preprocess_menu()
            case 2:
                train_menu()
            case 3:
                classify_menu()
            case 4:
                break
            case _:
                print("Not a valid choice! Try again. (1,2,3): ")

def preprocess_menu():
    while True:
        print("\nPreprocess Menu")
        print("preprocess strategies to add:")
        print("\t1) Noise Removal")
        print("\t2) Deduplication Removal")
        print("\t3) Translation")
        print("\t4) Back to main menu")

        df = load_data()

        choice = int(input("Enter your choice : "))
        match choice:
            case 1: 
                strategy = [NoiseRemovalStrategy()]
                df = preprocess_data(df, strategy,"AppGallery")
            case 2: 
                strategy = [DeduplicationStrategy()]
                df = preprocess_data(df, strategy,"AppGallery")
            case 3: 
                strategy = [TranslationStrategy()]
                df = preprocess_data(df, strategy,"AppGallery")
            case 4:
                break
            case _:
                print("Not a valid choice! Try again. (1,2,3): ")

def train_menu():
    while True:
        print("\nTrain Menu")
        print("Models you can train:")

        choice = moodel_selection()

        if "8" in choice.split():
            break

        if "7" in choice.split():
            print("Training all models...")
            train_choice(0b111111)  # All bits set
            continue

        choice_bitmask = choice_to_bitmask(choice)
        train_choice(choice_bitmask)
        main_menu()

def train_choice(choice_bitmask):
    df = load_data()
    X, df = get_embeddings(df)

    df = df.reset_index(drop=True)
    data = get_data_object(X, df)

    factory = ModelFactory() 

    for i in range(6):
        if choice_bitmask & (1 << i): 
            print(f"Training Model {i + 1}...")
            factory.create_model(1 << i)
            factory.train_evaluate(data, False) 
    
    print("Training complete!")

def classify_menu():
    while True:
        print("\nClassify Email Menu")
        print("1) Classify group of emails")
        print("2) Back to main menu")
        choice = int(input("Enter your choice : "))
        match choice:
            case 1:
                choice = moodel_selection()
                bitChoice = choice_to_bitmask(choice)
                classify_group(bitChoice)
            case 2:
                break
            case _:
                print("Not a valid choice! Try again. (1,2,3): ")

def valid_choices(choice):
    if not choice.strip():
        return False
    try:
        choices = list(map(int, choice.split()))
        return all(1 <= c <= 8 for c in choices)  # Only allow numbers 1-7
    except ValueError:
        return False

def choice_to_bitmask(choice_str):
    positions = list(map(int, choice_str.replace(" ", "")))

    bitmask = 0

    for pos in positions:
        bitmask |= (1 << (pos - 1))

    return bitmask

def moodel_selection():
    while True:
        print("1) Random Forest")
        print("2) Logistic Regression")
        print("3) SVM")
        print("4) Gradient Boosting")
        print("5) KNN")
        print("6) Naive Bayes")
        print("7) All models")
        print("8) Back to main menu")

        choice = input("Enter your choices (e.g type: 1 3 5 to choose models 1,3 and 5): ")

        if not valid_choices(choice):
            print("Invalid input! Please enter numbers between 1 and 7 separated by spaces. or 8 on its own")
            continue
    
        break

    return choice

def load_data():
    file_path = 'data/AppGallery_processed.csv'

    if os.path.exists(file_path):
        print("\nLoading from processed dataframe")
        df = load_processed_data()
    else:
        df = load_original_data()
        print("\nLoading from original dataframe")

    return df

def load_original_data():
    print("get_input_data()")
    data = pd.read_csv('data/AppGallery.csv')
    return data

def load_processed_data():
    data = pd.read_csv('data/AppGallery_processed.csv')
    return data

def load_unclassified_data():
    data = pd.read_csv('data/To_Classify.csv')
    return data

def get_embeddings(df:pd.DataFrame):
    vectorizer_path = 'data/vectorizer.pkl'
    
    if Config.INTERACTION_CONTENT in df.columns:
        valid_indices = df[Config.INTERACTION_CONTENT].notnull()
        df = df[valid_indices]
    else:
        raise KeyError(f"Column {Config.INTERACTION_CONTENT} not found in dataframe.")

    if df.empty:
        raise ValueError("No valid rows in dataframe to process.")
    
    if os.path.exists('data/vectorizer.pkl'):
        with open('data/vectorizer.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)
        X = loaded_vectorizer.transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
    else:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
        with open('data/vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)

    print(f"Embeddings shape: {X.shape}, DataFrame shape: {df.shape}")
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame, prediction: bool = False):
    return Data(X, df, prediction)

def classify_group(choice_bitmask):
    df = load_unclassified_data()

    strategy = [NoiseRemovalStrategy()]
    df = preprocess_data(df, strategy, "To_Classify")
                
    strategy = [DeduplicationStrategy()]
    df = preprocess_data(df, strategy, "To_Classify")
    
    X, df = get_embeddings(df)
    df = df.reset_index(drop=True)
    data = get_data_object(X, df, prediction=True)

    factory = ModelFactory()

    for i in range(6):
        if choice_bitmask & (1 << i): 
            print(f"Evaluating Emails {i + 1}...")
            factory.create_model(1 << i)
            pred = factory.predict(data)
            print(pred)
