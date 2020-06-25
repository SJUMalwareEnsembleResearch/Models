#Random Forest Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


dataset = pd.read_csv('all_data.csv')
X = dataset.iloc[:, 33:].values
Y = dataset.iloc[:, 0].values
print(X)
print(Y)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)



# Not needed for Random Forest
# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0) #Might need to change n_estimators
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJK"],
                  columns = [i for i in "ABCDEFGHIJK"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

#Overall Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_train, classifier.predict(X_train))
accuracy_score(Y_test, Y_pred)

#Precision and Recall
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average ='micro')
precision_recall_fscore_support(Y_test, Y_pred, average ='macro')
precision_recall_fscore_support(Y_test, Y_pred, average ='weighted')








