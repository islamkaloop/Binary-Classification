# load libs
import sys
import pickle
import pandas as pd
import os

features = []

# load data to convert the input to nums
tr_data = pd.read_csv(os.getcwd() + '/training.csv',";")
val_data = pd.read_csv(os.getcwd() + '/validation.csv',";")
tr_data = tr_data.fillna('missing')
val_data = val_data.fillna('missing')
all_data = tr_data.append(val_data, ignore_index = True) 

def creatdatadic(dataWithAllUniqueValues):
  unToken = list(set(dataWithAllUniqueValues.values))
  token2idx = {c: i for i, c in enumerate(unToken)}
  return token2idx

# get the features from the user
print("please make sure that the type of each feature is equal to the data type in the data set")
print("as the code will not hundel it")

for i in all_data:
    if(i == 'classLabel'):
        break
    x=input("Please enter the "+str(i)+" feature its type "+str(all_data[i].dtypes)+" > ")
    token2idx = creatdatadic(all_data[i])
    while (x==""):
        x=input("Please enter the "+str(i)+"th feature > ")
    if (not (x in token2idx)):
        features.append(len(token2idx))
    else:
        features.append(token2idx[x])

print("\nthe feature you entered is after convert it to nums")
print(features)

filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([features])
print('\nthe output is :')
if (result == 0):
    print("no.")
else:
    print("yes.")