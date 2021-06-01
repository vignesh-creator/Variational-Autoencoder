import torch
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
l_rate = 0.001
batch_size = 256
epochs = 15

#importing data
train_set = datasets.MNIST('~/torch_datasets', train=True, download=False)
test_set = datasets.MNIST('~/torch_datasets', train=False, download=False)
train_img = train_set.data.numpy()
test_img = test_set.data.numpy()
train_label = train_set.targets.numpy()
test_label = test_set.targets.numpy()

train_label = torch.Tensor(train_label)
test_label = torch.Tensor(test_label)

train_images = torch.Tensor(train_img).view(train_img.shape[0],1,28,28).to(device)
train_label = F.one_hot(train_label.to(torch.int64),10).to(device)
test_images = torch.Tensor(test_img).view(test_img.shape[0],1,28,28).to(device)
test_label = F.one_hot(test_label.to(torch.int64),10).to(device)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(794,512).to(device),
                nn.ReLU(),
                nn.Linear(512,256).to(device),
                nn.ReLU(),
                nn.Linear(256,128).to(device),
                nn.ReLU(),
                nn.Linear(128,64).to(device),
                nn.ReLU()
        )
        self.decoder = nn.Sequential(
                nn.Linear(32,64).to(device),
                nn.ReLU(),
                nn.Linear(64,128).to(device),
                nn.ReLU(),
                nn.Linear(128,256).to(device),
                nn.ReLU(),
                nn.Linear(256,512).to(device),
                nn.ReLU(),
                nn.Linear(512,784).to(device),
                nn.ReLU()
        )
    
    def forward(self,x,x_label):
        x=x.view(x.shape[0],-1)
        input_val=x
        x=torch.cat((x,x_label),1)
        x=self.encoder(x)
        mean,std=torch.chunk(x,2,dim=1)
        x = mean + torch.randn_like(std).to(device)*std
        x=self.decoder(x)
        return input_val,x,mean,std
# AutoEncoder
AutoEncoder = AE()

#optimiser
import torch.optim as optim
optimiser = torch.optim.Adam(AutoEncoder.parameters(), lr=0.005)
loss_list = []

#loss_function

def variational_loss(output,X_in,mean,std):
    loss_function = nn.MSELoss()
    loss_by_function=loss_function(output,X_in)
    kl_loss= -0.005*torch.sum(1+torch.log(torch.pow(std,2)+1e-10)-torch.pow(std,2)-torch.pow(mean,2))
    total_loss=loss_by_function+kl_loss
    return total_loss


def train(X,X_label):
    for epoch in range(0,epochs):
        cost = 0
        batch=torch.randperm(X.shape[0])
        for i in range(0, X.shape[0],batch_size):
            input_value,output,mean,std = AutoEncoder(X[batch[i:i+batch_size]],X_label[batch[i:i+batch_size]])
            optimiser.zero_grad()
            loss=variational_loss(output,input_value,mean,std)
            cost = cost+loss.item() 
            loss.backward()
            optimiser.step()
        loss_avg = cost / X.shape[0]
        loss_list.append(loss_avg)
        print("For iteration: ", epoch+1, " the loss is :", loss_avg)
    return loss_list


#plotting of loss curve
import matplotlib.pyplot as plt
train_loss= train(train_images.to(device),train_label.to(device))
plt.plot(loss_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

#test loss
def test(X,X_lab):
    with torch.no_grad():
        cost=0
        batch=torch.randperm(X.shape[0])
        for i in range(0, X.shape[0],batch_size):
            test_input,test_output,mean,std=AutoEncoder(X[batch[i:i+batch_size]].to(device),X_lab[batch[i:i+batch_size]].to(device))
            loss=variational_loss(test_output,test_input,mean,std)
            cost=cost+loss.item()
        print(cost/X.shape[0])
print("Test loss")
test(test_images,test_label)


#Normal Image
fig, axes = plt.subplots(1,2)
axes[0].imshow(test_img[14],cmap="gray")
axes[1].imshow(test_img[10],cmap="gray")

plt.show()

output_imgs=AutoEncoder(test_images[0:40].to(device),test_label[0:40].to(device))
output_img=(output_imgs[1].to(torch.device('cpu')).detach().numpy()).reshape(40,28,28)

fig, axes = plt.subplots(1,2)
axes[0].imshow(output_img[14],cmap="gray")
axes[1].imshow(output_img[10],cmap="gray")
plt.show()