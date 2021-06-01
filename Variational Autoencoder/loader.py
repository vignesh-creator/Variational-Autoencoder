import torch
from torchvision import transforms,datasets,models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = datasets.MNIST('~/torch_datasets', train=True, download=False)
test_set = datasets.MNIST('~/torch_datasets', train=False, download=False)
train_img = train_set.data.numpy()
test_img = test_set.data.numpy()
train_label = train_set.targets.numpy()
test_label = test_set.targets.numpy()


def train_loader_fn():
    train_images = torch.Tensor(train_img).view(train_img.shape[0],1,28,28).to(device)
    return train_images
def test_loader_fn():
    test_images = torch.Tensor(test_img).view(test_img.shape[0],1,28,28).to(device)
    return test_images






train_label=torch.Tensor(train_label)
test_label=torch.Tensor(test_label)
train_img = train_img.reshape(-1, 28, 28,1)