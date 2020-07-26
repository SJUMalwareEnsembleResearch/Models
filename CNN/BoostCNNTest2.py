import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('all_data2_new.csv')
print(df.shape)
df

#Getting X and Y Data
X = df.iloc[:, 34:]
Y = df.iloc[:, 1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 23)
print(X_train.shape)
print(X_test.shape)



file = "/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting/Test/y_pred_boosted9.sav"
file2 = "/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/CNN/Boosting/Train/y_pred_boosted9.sav"
Y_pred = pickle.load(open(file, 'rb'))
Y_pred_Train = pickle.load(open(file2, 'rb'))

from sklearn.metrics import accuracy_score
accuracy_score(Y_train, Y_pred_Train)
print(Y_test.shape)
print(Y_pred)
print(Y_test)
accuracy_score(Y_test, Y_pred)

from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(Y_train, Y_pred_Train)
balanced_accuracy_score(Y_test, Y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average ='micro')
precision_recall_fscore_support(Y_test, Y_pred, average ='macro')
precision_recall_fscore_support(Y_test, Y_pred, average ='weighted')

from collections import Counter
from sklearn import metrics
mapping = Counter(Y_pred)
#print(Counter(y_test))
mapping = dict(sorted(mapping.items()))
#--- 259.12324500083923 seconds ---

label_map = {"0":"ADLOAD","1":"AGENT","2":"ALLAPLE_A","3":"BHO","4":"BIFROSE","5":"CEEINJECT","6":"CYCBOT_G","7":"FAKEREAN","8":"HOTBAR","9":"INJECTOR","10":"ONLINEGAMES","11":"RENOS","12":"RIMECUD_A","13":"SMALL","14":"TOGA_RFN","15":"VB","16":"VBINJECT","17":"VOBFUS", "18":"VUNDO","19":"WINWEBSEC","20":"ZBOT"  }

#print(y_test)

def write_cm(cm):
    file = open("/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/cm/boostcmm.txt","w")
    for y in range(0, 21):
        for x in range(0, 21):
            string = (str(x) + " " + str(y) + " "+ str(round(cm[y][x],4)))
            file.write(string + "\n")
    file.close()

def plot_confusion_matrix(y_true,y_predicted):
    cm = metrics.confusion_matrix(y_true, y_predicted)
    l = list(cm)
    #print(l)
    s = 0
    for array in l:
        for value in array:
            s += value
    ooga = []
    counter = 0
    for array in l:
        array = list(array)
        array = [round(x /mapping[counter],3)  for x in array]
        ooga.append(array)
        counter += 1

    print(ooga)


    #cm = list((cm.T / cm.astype(np.float).sum(axis=1)).T)


    write_cm(ooga)
    #print ("Plotting the Confusion Matrix")


    labels = list(label_map.values())


    df_cm = pd.DataFrame(ooga,index = labels,columns = labels)
    fig = plt.figure(figsize=(20,10))
    ax = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

    plt.yticks([0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5], labels,va='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
    plt.show()
    plt.close()

plot_confusion_matrix(Y_test, Y_pred)
