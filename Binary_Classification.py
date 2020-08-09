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

# preprocess the data to make all data as nums
print("preprocess the data...")
def mapDataToIntType(data,dataWithAllUniqueValues):
  unToken = list(set(dataWithAllUniqueValues.values))
  token2idx = {c: i for i, c in enumerate(unToken)}

  X = [token2idx[w] for w in data]
  return X

def getAllFeatures(data, alldata):
  data = data[data.columns[0:len(data.columns)-1]]
  feature = []
  for i in data:
    feature.append(mapDataToIntType(data[i],all_data[i]))
  features = []
  for i in range(len(feature[0])):
    f = []
    for s in feature:
      f = f + [s[i]]
    features.append(f)
  return features

classes = list(set(all_data["classLabel"].values))
class2idx = {c: i for i, c in enumerate(classes)}
idx2class = {i: w for w, i in class2idx.items()}

y_tr = [class2idx[c] for c in tr_data["classLabel"]]
y_val = [class2idx[c] for c in val_data["classLabel"]]

tr_features = getAllFeatures(tr_data,all_data)
val_features = getAllFeatures(val_data,all_data)

input("Press Enter to continue...")
