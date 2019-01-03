'''
    The entry into your code. This file should include a training function and an evaluation function.
'''
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import GestureDataSet
from model import ConvolutionalNeuralNetwork

import matplotlib.pyplot as plt

feat_train = np.load('data/train_data.npy')
feat_valid = np.load('data/val_data.npy')
label_train = np.load('data/train_labels.npy')
label_valid = np.load('data/val_labels.npy')

#feat_train = np.delete(feat_train, [2,3,4,5], axis = 1)
#feat_valid = np.delete(feat_valid, [2,3,4,5], axis = 1)

print(feat_train.shape, feat_valid.shape)

np.random.seed(0)
torch.manual_seed(0)


def load_data(batch_size):

    train_dataset = GestureDataSet(feat_train,label_train)
    val_dataset = GestureDataSet(feat_valid,label_valid)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader

def load_model(lr):

    model = ConvolutionalNeuralNetwork(feat_train.shape[1])
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    return model, loss_fnc, optimizer

def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 3.6 YOUR CODE HERE
    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        prediction = model(feats.float())
        corr = torch.argmax(prediction, dim=1) == torch.argmax(label,dim=1)
        total_corr += int(corr.sum())
    ######
    #print(len(val_loader.dataset))
    return float(total_corr)/len(val_loader.dataset)

def test(model):

    instances = np.load('test_data.npy')

    normalized_data = np.zeros(instances.shape)

    averages = (np.average(instances, axis=1)).reshape(1170, 1, 6)
    stdevs = (np.std(instances, axis=1)).reshape(1170, 1, 6)

    normalized_data = (instances-averages)/stdevs
    normalized_data = normalized_data.transpose((0,2,1))

    test_dataset = GestureDataSet(normalized_data, np.zeros([normalized_data.shape[0], 26]))
    test_loader = DataLoader(test_dataset, batch_size=normalized_data.shape[0], shuffle=False)


    for i, batch in enumerate(test_loader):
        feats, label = batch
        feats = feats.float()
        preds = model(feats)

    predictions = []
    for i in range(preds.shape[0]):
        predictions.append(torch.argmax(preds[i]))

    np.savetxt('predictions.txt', predictions)

def train():
    bs = 32
    eval_every = 100
    eps = 35
    lr = 0.01

    train_loader, val_loader = load_data(bs)
    model, loss_fnc, optimizer = load_model(lr)

    t=0
    steps = []
    train_steps = []
    val_acc = []
    train_acc = []
    max_val = 0
    best_ep = 0
    print(type(train_loader))
    for epoch in range(eps):
        accum_loss = 0
        tot_corr = 0

        for i, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()
            predictions = model(feats.float())
            normallabels = torch.argmax(label,dim = 1)
            #print(normallabels)
            #print(predictions.shape)
            batch_loss = loss_fnc(input=predictions, target = normallabels.long())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            #print(torch.argmax(predictions,dim=1).shape,torch.argmax(label,dim=1).shape)
            corr = torch.argmax(predictions, dim=1) == torch.argmax(label,dim=1)
            tot_corr += int(corr.sum())

            if (t+1)% eval_every == 0:
                valid_acc = evaluate(model,val_loader)
                #training_acc = evaluate(model,train_loader)
                print("Epoch: {}, Step {} | Loss : {}| Test acc:{}".format(epoch+1, t+1, accum_loss/100, valid_acc))
                accum_loss = 0
                val_acc.append(valid_acc)
                #train_acc.append(training_acc)
                steps.append(t+1)

            t = t+1

        valid_acc = evaluate(model,val_loader)
        if valid_acc > max_val:
            max_val = valid_acc
            best_ep = epoch
        training_acc = evaluate(model,train_loader)
        print("Epoch: {}, Step {} | Training acc: {}| Test acc:{}".format(epoch + 1, t + 1, training_acc, valid_acc))
        val_acc.append(valid_acc)
        train_acc.append(training_acc)
        steps.append(t + 1)
        train_steps.append(t+1)
    print('Best Epoch: {} | Accuracy: {}'.format(best_ep, max_val))
    print("Train acc: {}".format(float(tot_corr) / len(train_loader.dataset)))

    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    steps = np.array(steps)
    train_steps = np.array(train_steps)

    torch.save(model,'model.pt')

    plt.plot(steps,val_acc)
    plt.plot(train_steps, train_acc)

    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title("Accuracy for Learning Rate = {} | Batch Size = {} | Epochs = {}".format(lr, bs, eps))
    plt.legend(['Validation Accuracy','Training Accuracy'])
    plt.show()


if __name__ == "__main__":
    train()
    model = torch.load('model.pt')
    test(model)