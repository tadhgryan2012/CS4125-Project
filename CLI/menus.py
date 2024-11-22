from CLI.classification_process import *
from modelling.data_model import *
from modelling.ModelFactory import *

def main_menu():
    
    run = True
    
    while run:
        print("\nMain Menu")
        print("1) Train a model")
        print("2) Classify an email")
        print("3) Exit")
        try:
            choice = int(input("Enter your choice : "))
            match choice:
                case 1: 
                    train_menu()
                case 2:
                    classify_menu()
                case 3:
                    run = False
                case _:
                    print("Not a valid choice! Try again. (1,2,3): ")
        except ValueError:
            print("WTF")

def train_menu():
    run = True
    while run:
        print("\nTrain Menu")
        print("Models you can train:")

        choice = moodel_selection()

        if "7" in choice.split():
            run = False
            continue

        if "6" in choice.split():
            print("Training all models...")
            train_choice(0b11111)  # All bits set
            continue
        
        choice_bitmask = choice_to_bitmask(choice)
        train_choice(choice_bitmask)
        main_menu()

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
        return all(1 <= c <= 7 for c in choices)  # Only allow numbers 1-7
    except ValueError:
        return False

def choice_to_bitmask(choice_str):
    positions = list(map(int, choice_str.replace(" ", "")))

    bitmask = 0

    for pos in positions:
        bitmask |= (1 << (pos - 1))

    print(bitmask)

    return bitmask

def train_choice(choice_bitmask):

    data = prepare_data() 
    
    factory = ModelFactory() 

    for i in range(5):
        print(i)
        if choice_bitmask & (1 << i): 
            print(f"Training Model {i + 1}...")
            factory.create_model(1 << i)
            factory.train_evaluate(data) 
    
    print("Training complete!")

def moodel_selection():

    run = True
    while run:

        print("1) Model1 (<- add name)")
        print("2) Model2 (<- add name)")
        print("3) Model3 (<- add name)")
        print("4) Model4 (<- add name)")
        print("5) Model5 (<- add name)")
        print("6) All models")
        print("7) Back to main menu")

        choice = input("Enter your choices (e.g type: 1 3 5 to choose models 1,3 and 5): ")

        if not valid_choices(choice):
            print("Invalid input! Please enter numbers between 1 and 6 separated by spaces. or 7 on its own")
            continue
    
        run = False


    return choice


def prepare_data():

    df = pd.read_csv("data\AppGallery.csv")
    X = df[['Interaction content', 'Ticket Summary']].values  
    data = Data(X, df) 

    return data

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

