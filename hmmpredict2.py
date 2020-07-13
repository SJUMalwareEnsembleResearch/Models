import pickle
import numpy as np

families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

filename = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm_models/5_50_0.5/finalized_model.sav'
filename2 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm_models/5_50_0.5/X_test.sav'
filename3 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm_models/5_50_0.5/Y_test.sav'
filename4 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm_models/5_50_0.5/Y_pred.sav'
all_models = pickle.load(open(filename, 'rb'))
testX = pickle.load(open(filename2, 'rb'))
Y_test = pickle.load(open(filename3, 'rb'))
Y_pred = pickle.load(open(filename4, 'rb'))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#Balanced Accuracy
from sklearn.metrics import balanced_accuracy_score
print(balanced_accuracy_score(Y_test, Y_pred))

