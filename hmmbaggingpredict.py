import pickle
import numpy as np

families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

file = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/Bagging/finalized_model'
fileX = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/Bagging/X_test.sav'
fileY = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/Bagging/Y_test.sav'

X_test = pickle.load(open(fileX, 'rb'))
Y_test = pickle.load(open(fileY, 'rb'))
Y_pred = np.empty(0, dtype=np.int8)
for i in range(1,6): #change later
    for row in X_test:
        array = [0] * 21
        modelFile = file + str(i) + '.sav'
        all_models = pickle.load(open(modelFile, 'rb'))
        pred = predict(row, all_models)
        array[pred] += 1
    final_Pred = findMax(array)
    global Y_pred
    Y_pred = np.append(Y_pred, final_Pred)
    print(families[final_Pred])
        


def predict(row, all_models): #data is 2d np array
    bestScore = -9999999999
    best_model = 0
    count = -1
    for model in all_models:
        count += 1
        try:
            score = model.score(np.reshape(row, (-1, 1)))
            if score > bestScore:
                bestScore = score
                best_model = count
        except:
            continue
    return best_model

def findMax(array):
    bestScore = -1
    best_model = -1
    for i in range(20):
        if (array[i] > bestScore):
            bestScore = array[i]
            best_model = i
    return best_model
        
    
    
filename4 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/Bagging/Y_pred.sav'
# print(Y_pred)
pickle.dump(Y_pred, open(filename4, 'wb'))

from sklearn.metrics import accuracy_score
print("-------------------------")
print(accuracy_score(Y_test, Y_pred))