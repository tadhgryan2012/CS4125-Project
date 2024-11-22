import os
from CLI.classification_process import *
from modelling.data_model import *
from modelling.ModelFactory import *
from preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer

def main_menu():
    
    run = True
    
    while run:
        print("\nMain Menu")
        print("1) Preprocess the Data")
        print("2) Train a model")
        print("3) Classify an email")
        print("4) Exit")
        try:
            choice = int(input("Enter your choice : "))
            match choice:
                case 1: 
                    preprocess_menu()
                case 2:
                    train_menu()
                case 3:
                    classify_menu()
                case 4:
                    run = False
                case _:
                    print("Not a valid choice! Try again. (1,2,3): ")
        except ValueError:
            print("WTF")

def preprocess_menu():
    run = True
    while run:
        print("\nPreprocess Menu")
        print("preprocess strategies to add:")
        print("1) Noise Removal")
        print("2) Deduplication Removal")
        print("3) Translation")
        print("4) Back to main menu")

        df = load_data()

        choice = int(input("Enter your choice : "))
        match choice:
            case 1: 
                strategy = [NoiseRemovalStrategy()]
                df = preprocess_data(df, strategy)
            case 2: 
                strategy = [DeduplicationStrategy()]
                df = preprocess_data(df, strategy)
            case 3: 
                strategy = [TranslationStrategy()]
                df = preprocess_data(df, strategy)
            case 4:
                run = False
            case _:
                print("Not a valid choice! Try again. (1,2,3): ")


def train_menu():
    run = True
    while run:
        print("\nTrain Menu")
        print("Models you can train:")

        choice = moodel_selection()

        if "8" in choice.split():
            run = False
            continue

        if "7" in choice.split():
            print("Training all models...")
            train_choice(0b11111)  # All bits set
            continue
        
        choice_bitmask = choice_to_bitmask(choice)
        train_choice(choice_bitmask)
        main_menu()

def train_choice(choice_bitmask):

    df = load_data()

    X,df = get_embeddings(df)

    data = get_data_object(X,df)

    factory = ModelFactory() 

    for i in range(5):
        print(i)
        if choice_bitmask & (1 << i): 
            print(f"Training Model {i + 1}...")
            factory.create_model(1 << i)
            factory.train_evaluate(data) 
    
    print("Training complete!")

def classify_menu():
    print("\nClassify Email Menu")
    print("1) Start email classification process")
    print("2) Back to main menu")
    try:
        choice = int(input("Enter your choice : "))
        match choice:
            case 1: 
                create_df_from_input()
            case 2:
                main_menu()
            case _:
                print("Not a valid choice! Try again. (1,2,3): ")
    except ValueError:
            print("Invalid input! Please enter a number. (1,2,3)")



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

    print(bitmask)

    return bitmask



def moodel_selection():

    run = True
    while run:

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
    
        run = False


    return choice



def load_data():
    file_path = 'data/AppGallery_processed.csv'

    if os.path.exists(file_path):
        print("Loading from processed dataframe")
        df = load_processed_data()
    else:
        df = load_original_data()
        print("Loading from original dataframe")

    return df

def load_original_data():
    print("get_input_data()")
    data = pd.read_csv('data/AppGallery.csv')
    return data

def load_processed_data():
    data = pd.read_csv('data/AppGallery_processed.csv')
    return data

def get_embeddings(df:pd.DataFrame):
    vectorizer = TfidfVectorizer()

    # Generate embeddings only for non-null rows
    valid_indices = df[Config.INTERACTION_CONTENT].notnull()
    df = df[valid_indices]
    X = vectorizer.fit_transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
    
    print(f"Embeddings shape: {X.shape}, DataFrame shape: {df.shape}")
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)



# bullshit for testing
def train_model1():
    print("model 1 trained!!!!")

def train_model2():
    print("model 3 trained!!!!")

def train_model3():
    print("model 3 trained!!!!")

def train_model4():
    print("model 4 trained!!!!")

def train_model5():
    print("model 5 trained!!!!")

