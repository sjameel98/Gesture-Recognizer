'''
    Visualize some basic statistics of our dataset.
'''
import numpy as np
import matplotlib.pyplot as plt

instances  = np.load('data/instances.npy')
labels = np.load('data/labels.npy')

#avgs = np.average(instances, axis = 1)
#avgs = np.reshape(avgs,(5590,1,6))

#inst_vars = np.std(instances, axis = 1)*99
#inst_vars = np.reshape(inst_vars, (5590,1,6))

#counts = 0
all_letter_avgs = np.zeros((26,215,100,6))
#all_letter_vars = np.zeros((26,43,6))
appended_count = np.zeros(26,)

for i in range(5590):
    letterindex = (np.where(labels[i] == 1))[0][0]
    #if (letterindex == 0):
    #    counts = counts+1
    all_letter_avgs[letterindex][int(appended_count[letterindex])] = instances[i]
    #print(letterindex, appended_count[letterindex])
    appended_count[letterindex] = appended_count[letterindex]+1

#all_letter_avgs = all_letter_avgs/counts
avgs = np.average(all_letter_avgs, axis = (1,2))
stdevs = np.std(all_letter_avgs, axis = (1,2))


entry = 1
index = np.arange(6)
alphabet = ['a','b','c']
for gest in range(3):
    plt.subplot(3,1,entry)
    plt.bar(index,avgs[gest].squeeze(),0.5,yerr = stdevs[gest].squeeze())
    entry = entry+1
    plt.title(alphabet[gest])
plt.show()
print(stdevs[0])
