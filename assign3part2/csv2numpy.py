'''
    Save the data in the .csv file, save as a .npy file in ./data
'''
import numpy as np
instances = np.zeros((5590,100,6))
labels = np.zeros((5590,26))
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
entry = 0
for stud in range(43):
    for letter in alphabet:
        for j in range(5):
            labels[entry][alphabet.index(letter)] = 1
            print(entry)
            filename = "data/unnamed_train_data/student"+str(stud)+'/'+letter+'_'+str(j+1)+'.csv'
            matrix = np.loadtxt(filename, delimiter=",")
            instances[entry] = matrix[:,1:]
            #print('student'+str(stud)+'/'+letter+'_'+str(j+1))
            #print(matrix.shape)
            entry = entry+1
print(instances[entry-1])
print(labels[entry-1])
print(labels[entry-1][25])
np.save("data/instances.npy",instances)
np.save("data/labels.npy",labels)

