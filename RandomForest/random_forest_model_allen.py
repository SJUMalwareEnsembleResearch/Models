import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

df = pd.read_csv('all_data.csv')

df.Family = df.Family.replace({"CEEINJECT": 0})
df.Family = df.Family.replace({"FAKEREAN": 1})
df.Family = df.Family.replace({"LOLYDA_BF": 2})
df.Family = df.Family.replace({"ONLINEGAMES": 3})
df.Family = df.Family.replace({"RENOS": 4})
df.Family = df.Family.replace({"STARTPAGE": 5})
df.Family = df.Family.replace({"VB": 6})
df.Family = df.Family.replace({"VBINJECT": 7})
df.Family = df.Family.replace({"VOBFUS": 8})
df.Family = df.Family.replace({"WINWEBSEC": 9})
df.Family = df.Family.replace({"ZBOT": 10})

df = df.loc[:, df.columns != 'Total Opcodes']

for i in range(31):
    df = df.drop(df.columns[1], axis=1)



opcode_sequence = (df.drop(df.columns[0], axis=1))
opcode_sequence = np.asarray(opcode_sequence)


labels = np.asarray(df[['Family']].copy())



X_train, X_test, y_train, y_test = train_test_split(opcode_sequence, labels, test_size=0.1, random_state=42)


rf = RandomForestClassifier(max_depth=50, random_state=0)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_accuracy =  metrics.accuracy_score(y_test, rf_preds)
print(rf_accuracy)


#n_estimators = [500, 800, 1500, 2500, 5000]
n_estimators = [100]
max_features = ['auto', 'sqrt','log2']
#max_depth = [10, 20, 30, 40, 50, 60, 70, 80]
max_depth = [50, 60, 70]
max_depth.append(None)
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15]




##RESUTS
#n estimators: 100
  #
#n estimators: 150
  #



grid_param = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf': min_samples_leaf}

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
RFR = RandomForestRegressor(random_state = 1)
RFR_random = RandomizedSearchCV(estimator = RFR, param_distributions = grid_param, n_iter = 500, cv =5, verbose = 2, random_state= 42, n_jobs = -1)
RFR_random.fit(X_train, y_train)
print(RFR_random.best_params_)





'''
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



class_names = ['a']
#plot_confusion_matrix(labels_test, rf_preds, classes=class_names, title='Confusion matrix, without normalization')

#plot_confusion_matrix(labels_test, rf_preds,classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()
'''