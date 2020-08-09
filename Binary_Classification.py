print("import libraries...")
# import libs
import pandas as pd
import os

input("Press Enter to continue...")
# load the data
print("load the data...")
tr_data = pd.read_csv(os.getcwd() + '/training.csv',";")
val_data = pd.read_csv(os.getcwd() + '/validation.csv',";")
tr_data = tr_data.fillna('missing')
val_data = val_data.fillna('missing')
all_data = tr_data.append(val_data, ignore_index = True) 
input("Press Enter to continue...")