from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np

DATA_PATH_X = '../data/cll/filtered_8d_1p_x_list.pkl'
DATA_PATH_Y = '../data/cll/y_1p_list.pkl'
SAVE_PATH = '../data/cll/8d'
random_state = 123

dev_size = .32
with open(DATA_PATH_X, 'rb') as f:
    x_list = pickle.load(f)

with open(DATA_PATH_Y, 'rb') as f:
    y_list = pickle.load(f)
#Dev set contains small amount of data to tune params, val set is used for cross-validation
x_val, x_dev, y_val, y_dev = train_test_split(x_list, y_list, test_size=dev_size, random_state=random_state, stratify= y_list)

print(len(x_val), len(x_dev), len(y_val), len(y_dev))
print(x_val[0].shape)
print(y_val)
print('Num_positive in dev: %d, Num negative in dev: %d' %(sum(y_dev), len(y_dev) - sum(y_dev)))

with open(os.path.join(SAVE_PATH, 'x_val_8d_1p.pkl'),  'wb') as f:
    pickle.dump(x_val, f)
with open(os.path.join(SAVE_PATH, 'x_dev_8d_1p.pkl'),  'wb') as f:
    pickle.dump(x_dev, f)
with open(os.path.join(SAVE_PATH, 'y_val_8d_1p.pkl'),  'wb') as f:
    pickle.dump(y_val, f)
with open(os.path.join(SAVE_PATH, 'y_dev_8d_1p.pkl'),  'wb') as f:
    pickle.dump(y_dev, f)
