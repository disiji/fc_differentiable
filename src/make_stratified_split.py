from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np

DATA_PATH_X = '../data/cll/filtered_8d_1p_x_list.pkl'
DATA_PATH_Y = '../data/cll/y_1p_list.pkl'
SAVE_PATH = '../data/cll/8d'
random_state = 123
raw_data_dir = '../data/cll/PB1_whole_mqian'

sorted_sample_ids = []
for filename in sorted(os.listdir(raw_data_dir)):
    sample_id = int(filename.split('_')[3])
    sorted_sample_ids.append(sample_id)
sorted_sample_ids = np.array(sorted_sample_ids)
print(sorted_sample_ids)

dev_size = .32
with open(DATA_PATH_X, 'rb') as f:
    x_list = pickle.load(f)
x_list = [np.hstack([x, sorted_sample_ids[idx] * np.ones([x.shape[0], 1])]) for idx, x in enumerate(x_list)]

with open(DATA_PATH_Y, 'rb') as f:
    y_list = pickle.load(f)
#Dev set contains small amount of data to tune params, val set is used for cross-validation
x_val, x_dev, y_val, y_dev = train_test_split(x_list, y_list, test_size=dev_size, random_state=random_state, stratify= y_list)

dev_labels_after_splitting = [x[0, -1] for x in x_dev]
val_labels_after_splitting = [x[0, -1] for x in x_val]
print(dev_labels_after_splitting)
print(val_labels_after_splitting)
x_val = [x[:, 0:x.shape[1] - 1] for x in x_val]
x_dev = [x[:, 0:x.shape[1] - 1] for x in x_dev]

print('double check shape looks right', x_val[0][0, :])

print(len(x_val), len(x_dev), len(y_val), len(y_dev))
print('first val has %d cells, and first dev has %d cells'%(x_val[0].shape[0], x_dev[0].shape[0]))
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
