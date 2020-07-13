from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

dataset = pd.read_csv('all_data2_new.csv')
X = dataset.iloc[0:130, 34:].values
Y = dataset.iloc[0:130, 1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
print(X_test.shape)

Y_pred = np.empty(0)
def predict(data): #data is 2d np array
    for row in data:
        max_prob = -9999999999
        best_model = 0
        count=0
        for model in all_models:
            score = model.score(np.reshape(row, (-1, 2)))
            if score > max_prob:
                best_model = count
            count += 1
        Y_pred = np.append(Y_pred, count)
        print(families[count])

# Displaying results
model = hmm.MultinomialHMM(n_components=5, n_iter=100, tol=0.5)
model.fit(X_train)
print("done training")
for row in X_test:
    row = np.reshape(row, (-1, 1))
    print(model.score(row))


