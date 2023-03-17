import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self,num_classes=4):

        super().__init__()
            
            #output size after convolution filter
            #Input shape= (64,3,200,200)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=12, kernel_size=3,stride=1,padding=1)
            #shape= (64,12,200,200)

        self.bn1 = nn.BatchNorm2d(num_features=12)
            #shape= (64,12,200,200)
            
        self.relu1 = nn.ReLU()
            #shape= (64,12,200,200)

        self.pool=nn.MaxPool2d(kernel_size=2)
            #Reduce the image size by a factor of 2
            #shape= (64,12,100,100)

        self.conv2=nn.Conv2d(in_channels=12,out_channels=20, kernel_size=3,stride=1,padding=1)
            #shape= (64,20,100,100)

        self.relu2 = nn.ReLU()
            #shape= (64,20,100,100)

        self.conv3=nn.Conv2d(in_channels=20,out_channels=32, kernel_size=3,stride=1,padding=1)
            #shape= (64,32,100,100)

        self.bn3=nn.BatchNorm2d(num_features=32)
            #shape= (64,32,100,100)

        self.relu3 = nn.ReLU()
            #shape= (64,32,100,100)

        self.fc = nn.Linear(in_features=32*100*100, out_features=4)

            #feed forward function

    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)
        
        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        #above output will be in matrix form, with shape of (64,32,100,100)

        output=output.view(-1,32*100*100)

        output=self.fc(output)

        return output
    
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)
        
        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        #above output will be in matrix form, with shape of (64,32,100,100)

        output=output.view(-1,32*100*100)

        output=self.fc(output)

        return output
