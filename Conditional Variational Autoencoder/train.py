import torch
from torchvision import transforms,datasets,models
import loader  #loads the MNIST dataset and necessary libraries
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
l_rate = 0.005
batch_size = 400
epochs = 5

# creates a convolution layer with 3x3 kernel size
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1).to(device)

#creates a transpose convolution layer with 3x3 kernel size
def convT3x3(in_channels, out_channels, stride=1,padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=padding).to(device)

# We are using a ResNet to train the data

# creating the residual blocks of the resnet model

# this is used as encoder residual block of the model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.subblock_1=nn.Sequential(
            conv3x3(in_channels, out_channels, stride).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU()
        )
        self.subblock_2=nn.Sequential(
            conv3x3(out_channels, out_channels).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU(),
        )
        self.downsample = downsample
    def forward(self, x):
        residual = x
        x = self.subblock_1(x).to(device)
        x = self.subblock_2(x).to(device)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = F.relu(x)        
        return x

block = ResidualBlock

# this is used as decoder residual block of the model
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,padding=1,upsample=None):
        super(DecoderResidualBlock, self).__init__()
        self.subblock_1=nn.Sequential(
            convT3x3(in_channels, out_channels, stride,padding).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU()
        )
        self.subblock_2=nn.Sequential(
            convT3x3(out_channels, out_channels).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU(),
        )
        self.upsample = upsample
    def forward(self, x):
        residual = x
        x = self.subblock_1(x).to(device)
        x = self.subblock_2(x).to(device)
        if self.upsample:
            residual = self.upsample(residual)
        x += residual
        x = F.relu(x)        
        return x
decoder_block=DecoderResidualBlock

# Creating the ResNet and inverse ResNet layers
class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels=16
        self.in_channels_decoder=32

        self.label_condition = nn.Linear(10,28*28)

        self.layer=nn.Sequential(
            conv3x3(2, 16).to(device),
            nn.BatchNorm2d(16).to(device),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block,16,2).to(device)
        self.layer2 = self.make_layer(block, 32,2,2).to(device)
        self.layer3 = self.make_layer(block, 64,2,2).to(device)
        self.max_pool = nn.MaxPool2d(7,1).to(device)
        self.fc = nn.Flatten()
        self.trans_layer1 = self.make_trans_layer(decoder_block,32,2).to(device)
        self.trans_layer2 = self.make_trans_layer(decoder_block, 16,2,2).to(device)
        self.trans_layer3 = self.make_trans_layer(decoder_block,3,2,2).to(device)
        self.trans_layer4 = nn.ConvTranspose2d(3,1,4,2,2).to(device)

    #making resnet layers
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    # making inverse resnet layer
    def make_trans_layer(self, decoder_block, out_channels, blocks, stride=1,padding=0):
        upsample = None
        if (stride != 1) or (self.in_channels_decoder != out_channels):
            upsample = nn.Sequential(
                convT3x3(self.in_channels_decoder, out_channels, stride=stride,padding=padding).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            )
        layers_dec = []
        layers_dec.append(decoder_block(self.in_channels_decoder, out_channels, stride,padding, upsample))
        self.in_channels_decoder = out_channels
        for i in range(1, blocks):
            layers_dec.append(decoder_block(out_channels, out_channels))
        return nn.Sequential(*layers_dec)

    def forward(self, x, y):

        y = self.label_condition(y)
        y = y.view(y.shape[0],1,28,28)
        x = torch.cat((x,y),1)


        # encoder part
        x = self.layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.max_pool(x)

        # latent vector generation
        latent = self.fc(x)
        mean,std=torch.chunk(latent,2,dim=1)

        #sampling of latent vecotr
        sample = mean + torch.randn_like(std)*std
        x=sample.view(sample.shape[0],32,1,1)
        x = self.trans_layer1(x)
        x = self.trans_layer2(x)
        x = self.trans_layer3(x)
        x = self.trans_layer4(x)
        return x,mean,std
AutoEncoder=ResNet()

#optimizer function
import torch.optim as optim
optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=l_rate)
loss_list = []

# calculation of variational loss
def variational_loss(output,X_in,mean,std):
    loss_function = nn.MSELoss()
    loss_by_function=loss_function(output,X_in)
    kl_loss= -(0.5/batch_size)*torch.sum(1+torch.log(torch.pow(std,2)+1e-10)-torch.pow(std,2)-torch.pow(mean,2))
    total_loss=loss_by_function+kl_loss
    return total_loss

#training function
def train(X,Y):
    for epoch in range(0,epochs):
        cost = 0
        batch=torch.randperm(X.shape[0]).to(device)
        for i in range(0, X.shape[0],batch_size):
            train_img = X[batch[i:i+batch_size]]
            train_lab = F.one_hot(Y[batch[i:i+batch_size]].long(),10).type(torch.float32)
            output,mean,std = AutoEncoder(train_img,train_lab)
            optimizer.zero_grad()
            loss=variational_loss(output,X[batch[i:i+batch_size]],mean,std)
            cost = cost+loss.item() 
            loss.backward()
            optimizer.step()
        loss_avg = cost / X.shape[0]
        loss_list.append(loss_avg)
        print("For iteration: ", epoch+1, " the loss is :", loss_avg)
    return loss_list


def test(X,Y):
    with torch.no_grad():
        cost=0
        batch=torch.randperm(X.shape[0])
        for i in range(0, X.shape[0],batch_size):
            test_img = X[batch[i:i+batch_size]]
            test_lab = F.one_hot(Y[batch[i:i+batch_size]].long(),10).type(torch.float32)
            output,mean,std=AutoEncoder(test_img,test_lab)
            loss=variational_loss(output,X[batch[i:i+batch_size]],mean,std)
            cost=cost+loss.item()
        print("Test set loss:",cost/X.shape[0])
    return output,test_img


def main():
    #loading train set images as tensors
    train_images = loader.train_loader_fn()
    train_label = loader.train_label
    # training the dataset
    train_loss = train(train_images,train_label)

    # plotting the cost function
    plt.plot(loss_list)
    plt.title("Loss curve")
    plt.ylabel('cost')
    plt.xlabel('epoch number')
    plt.show()

    # loading the test set of images
    test_images = loader.test_loader_fn()
    test_label = loader.test_label
    output_img,test_img = test(test_images,test_label)

    n = 10 # number of images that are to be displayed

    # n test images passed through variational autoencoder
    output_img=((output_img.to(torch.device('cpu'))).detach().numpy()).reshape(output_img.shape[0],28,28)
    output_img = output_img[:n]

    for i in range(0,n):
        axes = plt.subplot(2,n,i+1)
        plt.imshow(test_img[i].reshape(28,28),cmap="gray")
        axes.get_xaxis().set_visible(False) #removing axes
        axes.get_yaxis().set_visible(False)

        axes = plt.subplot(2,n,n+i+1)
        plt.imshow(output_img[i],cmap="gray")
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()


    



