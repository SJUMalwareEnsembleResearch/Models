import pickle
import numpy as np
filename = 'finalized_model.sav'

all_models = pickle.load(open(filename, 'rb'))
model = all_models[20]
print(model.score(np.reshape([1, 3, 3, 7, 3, 2, 10, 20, 29], (-1, 1))))