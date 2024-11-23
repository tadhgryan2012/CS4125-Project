from modelling.data_model import *
from modelling.ModelFactory import *
from CLI.timeFormatter import *
from CLI.menus import *
from preprocess.preprocess import *
import pandas as pd


email_components = []

columns = ['Ticket id','Interaction id','Interaction date','Mailbox','Ticket Summary','Interaction content','Innso TYPOLOGY_TICKET']

def create_df_from_input():

    print("If you do not have the information for a question, leave it blank and hit enter")
    
    current_col = int(input("Enter the ticket ID : "))
    email_components.append(current_col)

    current_col = int(input("Enter the interaction ID : "))
    email_components.append(current_col)

    current_col = str(input("Enter the interaction date (YYYY-MM-DD-hh-mm-ss) : "))
    current_col = excel_serial_date(current_col)
    email_components.append(current_col)

    current_col = input("Enter the mailbox that recieved the email : ")
    email_components.append(current_col)

    current_col = input("Enter the ticket summary : ")
    email_components.append(current_col)
    
    current_col = input("Enter the interaction content : ")
    email_components.append(current_col)

    df = pd.DataFrame(email_components, columns=columns)

    df = preprocess_data(df)
    return df


def classify_group(choice_bitmask):

    from CLI.menus import load_unclassified_data, get_embeddings, get_data_object

    df = load_unclassified_data()
    X, df = get_embeddings(df)
    df = df.reset_index(drop=True)
    data = get_data_object(X, df)

    strategy = [NoiseRemovalStrategy()]
    df = preprocess_data(df, strategy, "To_Classify")
                
    strategy = [DeduplicationStrategy()]
    df = preprocess_data(df, strategy, "To_Classify")
                
    #strategy = [TranslationStrategy()]
    #df = preprocess_data(df, strategy, "To_Classify")

    factory = ModelFactory()

    for i in range(6):
        if choice_bitmask & (1 << i): 
            print(f"Evaluating Emails {i + 1}...")
            factory.create_model(1 << i)
            print("HERE 1")
            pred = factory.predict(data)
            print(pred)
            



#Ticket id,Interaction id,Interaction date,Mailbox,Ticket Summary,Interaction content,Innso TYPOLOGY_TICKET
