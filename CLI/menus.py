import os
import pickle
from modelling.data_model import *
from modelling.ModelFactory import *
from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess.preprocess import preprocess_data
from preprocess.strategies import DeduplicationStrategy, NoiseRemovalStrategy, TranslationStrategy
from utils.Util import Util

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
        print("\t4) Do all")
        print("\t5) Perform Processes and Go Back")

        df = Util.get_data()

        choice = int(input("Enter your choice : "))
        strategy = []
        match choice:
            case 1: 
                print("Added NoiseRemovalStrategy to list")
                strategy.append(NoiseRemovalStrategy())
            case 2: 
                print("Added DeduplicationStrategy to list")
                strategy.append(DeduplicationStrategy())
            case 3: 
                print("Added TranslationStrategy to list")
                strategy.append(TranslationStrategy())
            case 4: 
                print("Adding all to list")
                strategy.append(DeduplicationStrategy())
                strategy.append(NoiseRemovalStrategy())
                strategy.append(TranslationStrategy())
                preprocess_data(df, strategy,"AppGallery")
                break
            case 5:
                print("Performing processes")
                if len(strategy) > 0:
                    preprocess_data(df, strategy,"AppGallery")
                break
            case _:
                print("Not a valid choice! Try again. (1,2,3): ")

def train_menu():
    print("\nTrain Menu")
    print("Models you can train:")
    
    choice = moodel_selection()
    
    if "8" in choice.split():
        return
    
    if "7" in choice.split():
        print("Training all models...")
        train_choice(0b111111)  # All bits set
    
    choice_bitmask = choice_to_bitmask(choice)
    train_choice(choice_bitmask)

def train_choice(choice_bitmask):
    df = Util.get_data()
    X, df = Util.get_embeddings(df)

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
    choice = moodel_selection()
    bitChoice = choice_to_bitmask(choice)
    classify_group(bitChoice)

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

def get_data_object(X: np.ndarray, df: pd.DataFrame, prediction: bool = False):
    return Data(X, df, prediction)

def classify_group(choice_bitmask):
    df = Util.get_data('data/To_Classify.csv')

    strategy = [DeduplicationStrategy(), NoiseRemovalStrategy(), TranslationStrategy()]
    df = preprocess_data(df, strategy, "To_Classify")
    
    df = Util.get_data('data/To_Classify_processed.csv')
    
    X, df = Util.get_embeddings(df)
    df = df.reset_index(drop=True)
    data = get_data_object(X, df, prediction=True)

    factory = ModelFactory()

    for i in range(6):
        if choice_bitmask & (1 << i): 
            print(f"Evaluating Emails {i + 1}...")
            factory.create_model(1 << i)
            pred = factory.predict(data)
            print(pred)
