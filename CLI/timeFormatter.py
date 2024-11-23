from datetime import datetime

def excel_serial_date(input_str):

    dt = datetime.strptime(input_str, "%Y-%m-%d-%H-%M-%S")
    
    # Excel's epoch is January 1, 1900
    excel_epoch = datetime(1900, 1, 1)
    
    # Calculate the number of days since Excel's epoch
    delta = dt - excel_epoch
    excel_date = delta.days + delta.seconds / 86400  # 86400 seconds in a day
    
    # Excel's leap year bug:
    # If the date is after February 28, 1900, we need to account for the false leap year assumption.
    if dt >= datetime(1900, 3, 1):  # Only add 1 if the date is after February 28, 1900
        excel_date += 1
    
    print(excel_date)
    return excel_date