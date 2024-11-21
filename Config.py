# This file contains some variable names you need to use in overall project. 
#For example, this will contain the name of dataframe columns we will working on each file
class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
    CLASS_COL = 'Type 1'  # Set to the broadest classification column
    GROUPED = 'Innso TYPOLOGY_TICKET '