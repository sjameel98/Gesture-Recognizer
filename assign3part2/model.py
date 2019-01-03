'''
    Write a model for gesture classification.
'''
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(6,16,kernel_size = 3, padding = 1, bias = True)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.activ1 = nn.LeakyReLU(0.1)
        self.mpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16,32,kernel_size = 3, padding = 1, bias = False)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.activ2 = nn.LeakyReLU(0.1)
        self.mpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32,64,kernel_size = 3, padding = 1, bias = False)
        self.conv3_bn = nn.BatchNorm1d(64)
        self.activ3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv1d(64,64,kernel_size = 3, padding = 1, bias = False)
        self.conv4_bn = nn.BatchNorm1d(64)
        self.activ4 = nn.LeakyReLU(0.1)

        self.mpool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(64,128,kernel_size = 3, padding = 1, bias = False)
        self.conv5_bn = nn.BatchNorm1d(128)
        self.activ5 = nn.LeakyReLU(0.1)

        self.conv6 = nn.Conv1d(128,128,kernel_size = 3, padding = 1)
        self.conv6_bn = nn.BatchNorm1d(128)
        self.activ6 = nn.LeakyReLU(0.1)

        self.mpool6 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(768,26)




    def forward(self,x):

        x = self.mpool1(self.activ1(self.conv1_bn(self.conv1(x))))

        x = self.mpool2(self.activ2(self.conv2_bn(self.conv2(x))))

        x = self.activ3(self.conv3_bn(self.conv3(x)))

        x = self.mpool4(self.activ4(self.conv4_bn(self.conv4(x))))

        x = self.activ5(self.conv5_bn(self.conv5(x)))

        x = self.mpool6(self.activ6(self.conv6_bn(self.conv6(x))))

        x = x.view(-1, x.shape[1] * x.shape[2])

        x = self.fc1(x)

        return x



    '''def __init__(self, input_size):
        super(ConvolutionalNeuralNetwork, self).__init__()

        #self.conv1 = nn.Conv1d(6, 9, 7, padding=2)  # increasing to 10 makes this worse
        self.conv11 = nn.Conv1d(6,12,3)
        self.conv12 = nn.Conv1d(12,12,3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(12, 12, 7, padding=2)  # decreasing to 9 made training accuracy better but probably because overfit
        # self.c2_bn = nn.BatchNorm1d(12)
        self.conv3 = nn.Conv1d(12, 15, 7, padding=2)  # increasing this to 16 worsens, same with 14
        self.c3_bn = nn.BatchNorm1d(15)  # keep this batch normalization

        self.fc1 = nn.Linear(150, 120)  # changing to 110,130 makes this suck
        self.fc2 = nn.Linear(120, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 26)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv12(F.tanh(self.conv11(x))))) #leaky relu on second tanh doesn't work well
        x = self.pool(F.tanh(self.conv2(x)))  # dropout is poopoo
        x = self.c3_bn(self.pool(F.tanh(self.conv3(x))))  # don't use dropout before beginning

        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.tanh(self.fc1(x))
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.1)  # increasing the 0.1 is bad
        x = self.fc3(x)
        # x = F.softmax(x, dim=1) remove softmax completely

        return x'''
    '''
    def __init__(self, input_size):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(6, 9, 7, padding=2) #increasing to 10 makes this worse
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(9, 12, 7, padding=2) #decreasing to 9 made training accuracy better but probably because overfit
        #self.c2_bn = nn.BatchNorm1d(12)
        self.conv3 = nn.Conv1d(12, 15, 7, padding=2) #increasing this to 16 worsens, same with 14
        self.c3_bn = nn.BatchNorm1d(15) #keep this batch normalization
        self.drop = nn.Dropout(p = 0.1)


        self.fc1 = nn.Linear(150, 120) #changing to 110,130 makes this suck
        self.fc2 = nn.Linear(120, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 26)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x))) #dropout is poopoo
        x = self.c3_bn(self.pool(F.tanh(self.conv3(x)))) #don't use dropout before beginning

        x = x.view(-1, x.shape[1]*x.shape[2])
        x = F.tanh(self.fc1(x))
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)),0.1) #increasing the 0.1 is bad
        x = self.fc3(x)
        #x = F.softmax(x, dim=1) remove softmax completely 


        return x
    '''

#0.7790697674418605 at epoch 27
#Epoch: 23 | Accuracy: 0.8014311270125224 at epoch 23 for last being relu...even to 100 epochs --> but still better than