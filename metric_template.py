# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:14:17 2020

@author: Gavin
"""

#Confusion Matrix
from sklearn.metrics import confusion_matrix
families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A' 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'LOLYDA_BF', 'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'STARTPAGE', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINTRIM_BX', 'WINWEBSEC', 'ZBOT']
cm = confusion_matrix(Y_test, Y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in families],
                  columns = [i for i in families])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

#Overall Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_train, classifier.predict(X_train))
accuracy_score(Y_test, Y_pred)

#Precision, Recall, F1Score
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(Y_test, Y_pred, average ='micro')
precision_recall_fscore_support(Y_test, Y_pred, average ='macro')