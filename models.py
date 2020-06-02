## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
         # first Max-pooling layer
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(p=0.1)
        
        # second convolutional layer
        self.conv2 = nn.Conv2d(32,64,3) 
        
        # second Max-pooling layer
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Third convolution layer
        self.conv3 = nn.Conv2d(64,128,3)
        
        # Third max pooling layer
        self.pool3 = torch.nn.MaxPool2d(2,2)
        self.dropout3 = nn.Dropout(p=0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2,2)
        self.dropout4 = nn.Dropout(p=0.4)
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2,2)
        self.dropout5 = nn.Dropout(p=0.5)
        
        # Fully connected layer
        self.fc1 = torch.nn.Linear(512*5*5, 1000)
        self.drop6 = nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(1000, 600)
        self.drop7 = nn.Dropout(p=0.4)
        self.fc3 = torch.nn.Linear(600, 136) 
        
  
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)
        

        
    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.relu(self.conv4(x))))
        x = self.dropout5(self.pool5(F.relu(self.conv5(x))))
        
      
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop6(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop7(x)
        
        x = self.fc3(x)
       
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
