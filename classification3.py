#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:00:46 2018

@author: yuhan_long
"""

import torch
import torchvision
from torchvision import transforms, utils,datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
#%%
path_train = '/home/yuhan_long/Documents/DogAndCat/train/'
path_test = '/home/yuhan_long/Documents/DogAndCat/test/'

transform = transforms.Compose(
        [transforms.Resize(64),
         transforms.CenterCrop(64),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.ImageFolder(
        root = path_train, 
        transform = transform
        )
trainloader = DataLoader(
        trainset, 
        batch_size = 4, 
        shuffle = True)

testset = datasets.ImageFolder(
        root = path_test,
        transform = transform
        )
#%%
testloader = DataLoader(
        testset,
        batch_size = 75,
        shuffle = True)
#%%
# build CNN 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(                 # input shape (3, 64, 64)
                nn.Conv2d(
                        in_channels = 3,
                        out_channels = 64,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1                 # padding = (kernel - 1) / 2
                        ),                          # output shape (64, 64, 64)
                nn.BatchNorm2d(64),
                nn.ReLU(),                          # activation
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 4)       # ouput shape (64, 16, 16)
                )
                
        self.conv2 = nn.Sequential(                 # -> (64, 16, 16)
                nn.Conv2d(64, 128, 3, 1, 1),        # shape (128, 16, 16)
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(4)
                )                    # shape (128, 4, 4)
        
        self.conv3 = nn.Sequential(                 # -> (128, 4, 4)
                nn.Conv2d(128, 256, 3, 1, 1),        # shape (256, 4, 4)
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(4)
                )                    # shape (256, 1, 1)
        
        self.fc = nn.Sequential(
#                nn.Linear(64 * 16 * 16, 4096),
#                nn.Linear(256, 128),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(256, 2)
#                nn.Linear(128, 2),
                )
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
#        x = self.conv3(x)
        x = x.view(x.size(0), -1)                   # flatten
        output = self.fc(x)
        return output, x                            # x for visualization
    
cnn = CNN()
cnn.cuda()
#print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.0001)
criteria = nn.CrossEntropyLoss()
#criteria = nn.MSELoss()

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
import matplotlib.pyplot as plt
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# training and testing
#%%
# Prepare test_img and test_label for visualization of Training process
for img, label in testloader:
    test_img = Variable(img, requires_grad=True).cuda()
    test_label = Variable(label).cuda()
    break

#%%
plt.ion()
for epoch in range(5):
    step = 0
    for img, label in trainloader:
        train_img = Variable(img, requires_grad=True).cuda()
        train_label = Variable(label).cuda()
        
        output, x = cnn(train_img)
        loss = criteria(output, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
#        # testing
        if step % 500 == 0:
            test_out, last_layer = cnn(test_img)
            pred_label = torch.max(test_out, 1)[1].data.squeeze().cpu().numpy() # one-hot
#           print(sum(pred_label == test_label.cpu().data.numpy()) )
#           break
            accuracy = sum(pred_label == test_label.cpu().data.numpy()) / float(test_label.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
            # visualization of last layer
#            if epoch > 8 and HAS_SK:
#                # Visualization of trained flatten layer
#                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                plot_only = 500
#                low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
#                labels = test_label.cpu().data.numpy()[:plot_only]
#                plot_with_labels(low_dim_embs, labels)
        if step == 3999:
            print('Epoch: ', epoch)
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            image = torchvision.utils.make_grid(test_img[:4].cpu().data)
            image = image.numpy().transpose((1, 2, 0))
            image = image*std + mean
            
            dict = {0:'cat',1:'dog'}
            print([dict[pred_label[i]] for i in range(4)])
            plt.imshow(image)
                
        step += 1
        
plt.ioff()
#%%
# print 10 predictions from test data
test_output, _ = cnn(test_img)
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
print(pred_y[50:60], 'prediction number')
print(test_label[50:60].data.cpu().numpy(), 'real number')   
#%%
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
image = torchvision.utils.make_grid(test_img[50:55].cpu().data)
image = image.numpy().transpose((1, 2, 0))
image = image*std + mean

dict = {0:'cat',1:'dog'}
print([dict[pred_y[50+i]] for i in range(5)])
plt.imshow(image)
#%% testing all testset
step = 0
for img, label in testloader:
    test_img = Variable(img, requires_grad=True).cuda()
    test_label = Variable(label).cuda()
    
    test_out1, _ = cnn(test_img)
    pred_out1 = torch.max(test_out1, 1)[1].data.cpu().numpy().squeeze()
    print("Batch: ", step + 1)
    print(pred_y[50:60], 'prediction number')
    print(test_label[50:60].data.cpu().numpy(), 'real number')  
    step += 1