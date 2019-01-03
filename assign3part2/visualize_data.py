'''
    Visualize some samples.
'''
import numpy as np
import matplotlib.pyplot as plt

alphabet = ['a', 'x']
entry = 1

for letter in alphabet:
    for stud in range(3):
        filename = "data/unnamed_train_data/student"+str(stud)+'/'+letter+'_4'+'.csv'
        matrix = np.loadtxt(filename, delimiter=",")
        plt.subplot(2,3,entry)
        for plot in range(6):
            plt.plot(matrix[:,0],matrix[:,plot+1])
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title(str(stud) + '_' + letter + '_4')
        plt.legend(['ax', 'ay', 'az', 'wx', 'wy', 'wz'])

        entry = entry+1

plt.show()