
#class Email:
#
 #   def __init__(self):
#
 #       ticket_id = 0
  #      interaction_id = 0
#
 #       date_time_unformatted = ""
  #      date_time_formatted = ""
#
 #       mailbox = ""
#
 #       ticket_summary = ""
  #      interaction_content = ""
   #     typology_ticket = ""

#email = Email()

#email.ticket_id = 1

email_components = []

def classificationProcess():


    print("If you do not have the information for a question, leave it blank and hit enter")
    
    current_col = input("Enter the ticket ID : ")
    email_components.append(current_col)

    current_col = input("Enter the interaction ID : ")
    email_components.append(current_col)

    current_col = input("Enter the interaction date (YYYY-MM-DD-hh-mm-ss) : ")
    email_components.append(current_col)

    current_col = input("Enter the mailbox that recieved the email : ")
    email_components.append(current_col)

    current_col = input("Enter the ticket summary : ")
    email_components.append(current_col)
    
    current_col = input("Enter the interaction content : ")
    email_components.append(current_col)


#Ticket id,Interaction id,Interaction date,Mailbox,Ticket Summary,Interaction content,Innso TYPOLOGY_TICKET
