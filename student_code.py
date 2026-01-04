# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 256, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 128, bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(128, num_classes, bias=True)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        shape_dict[1] = list(x.shape)          # [N, 6, 14, 14]

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        shape_dict[2] = list(x.shape)          # [N, 16, 5, 5]

        x = torch.flatten(x, 1)
        shape_dict[3] = list(x.shape)          # [N, 400]

        x = self.fc1(x)
        x = self.relu3(x)
        shape_dict[4] = list(x.shape)          # [N, 256]

        x = self.fc2(x)
        x = self.relu4(x)
        shape_dict[5] = list(x.shape)          # [N, 128]

        out = self.fc3(x)
        shape_dict[6] = list(out.shape)        # [N, 100]

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    model_params = total / 1e6
    
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

#test
if __name__ == "__main__":
    import torch
    model = LeNet()
    x = torch.randn(4, 3, 32, 32) 
    out, shape_dict = model(x)
    print("Output shape:", out.shape)
    print("Shape dict:", shape_dict)
