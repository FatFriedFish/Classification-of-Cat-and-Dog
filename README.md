# Classification-of-Cat-and-Dog
For this project, PyTorch was employed to build a Convolutional Neural Network, which is used to classify cats and dogs. The accuracy of the CNN can be 96% on testing set.
## Pre-Process
For training set, there are 10000 labeled figures for cats and dogs repectively; and there are 5000 un-labeled figures for testing set. 

```Python
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
testloader = DataLoader(
        testset,
        batch_size = 75,
        shuffle = True)
```
## Build CNN
For better performance, batch normalization and dropout was used to avoid overfitting.
```Python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(                 #
                nn.Conv2d(
                        in_channels = 3,
                        out_channels = 64,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1                 
                        ),                          
                nn.BatchNorm2d(64),
                nn.ReLU(),                          
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 4)       
                )
                
        self.conv2 = nn.Sequential(                 
                nn.Conv2d(64, 128, 3, 1, 1),        
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(4)
                )      
         self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(256, 2)
                )
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)                 
        output = self.fc(x)
        return output, x       
```
