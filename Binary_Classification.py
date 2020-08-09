# we use sklearn as it is just a machine learning
print("import libraries...")
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# import libs
import pandas as pd
import os

max_acc = 0
max_trained_model = None
max_trained_model_str = None
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

# I decide the machine learning algo based on this map https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# we predict a category
# it is supervised learning
# data less than 100K
# cheak lineaar SVC
print("train lineaar SVC model...")
SVC = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
SVC.fit(tr_features, y_tr)

y_pred = SVC.predict(val_features)
print("lineaar SVC results:")
acc = accuracy_score(y_val,y_pred)
print("accuracy_score = "+str(acc))
if(acc>max_acc):
  max_acc = acc
  max_trained_model = SVC
  max_trained_model_str = "lineaar SVC model"
print("classification_report \n"+str(classification_report(y_val,y_pred)))
input("Press Enter to continue...")
