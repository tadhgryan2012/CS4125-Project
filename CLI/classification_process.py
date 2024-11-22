from modelling.data_model import *
from modelling.ModelFactory import *
from CLI.timeFormatter import *
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


#Ticket id,Interaction id,Interaction date,Mailbox,Ticket Summary,Interaction content,Innso TYPOLOGY_TICKET
