'''
    Normalize the data, save as ./data/normalized_data.npy
'''
import numpy as np
from sklearn.model_selection import train_test_split

instances  = np.load('data/instances.npy')
print(instances[-1][-1])
labels = np.load('data/labels.npy')

normalized_data = np.zeros(instances.shape)

averages = (np.average(instances, axis = 1)).reshape(5590,1,6)
stdevs = (np.std(instances, axis = 1)).reshape(5590,1,6)

normalized_data = (instances-averages)/stdevs
#print(normalized_data[5500][10])

np.save('data/normalized_data.npy',normalized_data)

feat_train, feat_valid, label_train, label_valid = train_test_split(normalized_data, labels, test_size = 0.2, random_state = 0)
feat_train = feat_train.transpose((0,2,1))
feat_valid = feat_valid.transpose((0,2,1))


print(feat_train.shape, feat_valid.shape, label_train.shape, label_valid.shape)
np.save('data/train_data.npy', feat_train)
np.save('data/val_data.npy', feat_valid)
np.save('data/train_labels.npy', label_train)
np.save('data/val_labels', label_valid)