'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''
import torch.utils.data as data

class GestureDataSet(data.Dataset):

    def __init__(self, X, y):

        pass
        ######

        # 3.1 YOUR CODE HERE
        self.X = X
        self.y = y
        ######

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        pass
        ######

        # 3.1 YOUR CODE HERE
        features = self.X[index]
        label = self.y[index]
        return features, label
        ######