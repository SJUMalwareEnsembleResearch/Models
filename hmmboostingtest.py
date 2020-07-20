import pickle
import numpy as np

families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',
            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',
            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']

filename = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Boosting/finalized_model1.sav'
filename2 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/X_train.sav'
filename3 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Y_train.sav'
filename4 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Boosting/errors.sav'
filename5 = '/Users/gavinwong/Desktop/Repos/SJUMalwareEnsembleResearch/Models/hmm/Boosting/Y_pred.sav'
all_models = pickle.load(open(filename, 'rb'))
X = pickle.load(open(filename2, 'rb'))
Y = pickle.load(open(filename3, 'rb'))
print(X.shape)
Y_pred = np.empty(0, dtype=np.int8)
error = [np.empty((0,0), dtype = np.int8)] * 21
def predict(data): #data is 2d np array
    for row in data:
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
        global Y_pred
        Y_pred = np.append(Y_pred, best_model)
        print(families[best_model])

#Displaying results
predict(X)
# print(Y_pred)


from sklearn.metrics import accuracy_score
print("-------------------------")
print(accuracy_score(Y, Y_pred))
pickle.dump(Y_pred, open(filename5, 'wb'))

for i in range(7284):
    if(Y[i] != Y_pred[i]):
        error[i] = np.append(error[i], X[i])

pickle.dump(error, open(filename4, 'wb'))





