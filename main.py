# reference from below two scripts
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/CIFAR10_ResNet50.ipynb

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np

import json
from tqdm import tqdm

from model.resnet import *

# set transform
train_transform = transforms.Compose([
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Dataset
dataset_path = "./data"
train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)

# Set DataLoader
trainloader = DataLoader(train, batch_size=256)
testloader = DataLoader(test, batch_size=256)

# training fuction
def train(epochs):
    train_loss_by_epoch = []
    train_acc_by_epoch = []
    test_loss_by_epoch = []
    test_acc_by_epoch = []
    
    for epoch in range(epochs):
        train_loss_list = []
        train_acc_list = []
        print('='*50)
        print(f'==> {block.__name__} - Epoch : {epoch}')
        print('='*50)
        for idx, i in enumerate(tqdm(trainloader)):
            input, label = i
            input, label = input.cuda(), label.cuda()
            out = model(input)
            
            optimizer.zero_grad()
            loss = loss_fct(out, label)
            train_loss_list.append(loss.item())
            _, predicted = torch.max(out.data, 1)
            acc = (predicted == label).cpu().numpy().mean()
            train_acc_list.append(acc)
            
            loss.backward()
            optimizer.step()
            
        train_loss = np.array(train_loss_list).mean()
        train_loss_by_epoch.append(train_loss)
        train_acc = np.array(train_acc_list).mean()        
        train_acc_by_epoch.append(train_acc)
        print(f"train_loss: {train_loss}")
        print(f"train_acc: {train_acc}")
        
        scheduler.step(train_loss)        
        current_lr = scheduler.optimizer.param_groups[0]['lr']
    
        test_loss_list = []
        test_acc_list = []
        with torch.no_grad():
            for idx, i in enumerate(testloader):
                input, label  = i
                input, label = input.cuda(), label.cuda()
                out = model(input)
    
                loss = loss_fct(out, label)
                test_loss_list.append(loss.item())
    
                _, predicted = torch.max(out.data, 1)
                acc = (predicted == label).cpu().numpy().mean()
                test_acc_list.append(acc)
            
            test_loss = np.array(test_loss_list).mean()
            test_loss_by_epoch.append(test_loss)        
            test_acc = np.array(test_acc_list).mean()    
            test_acc_by_epoch.append(test_acc)
            print(f"test_loss: {test_loss}")
            print(f"test_acc: {test_acc}")        
    
    
    result = {
        'train_loss': train_loss_by_epoch,
        'train_acc': train_acc_by_epoch,
        'test_loss': test_loss_by_epoch,
        'test_acc': test_acc_by_epoch
    }

    return result

# Set model
num_layers = [2, 2, 2, 2]
block_list = [Block, Bottleneck, PreactivationBlock, PreactivationBottleneck]

# Set Epoch
epochs = 25

# Set optimizer parameter
lr = 0.1
momentum = 0.9
weight_decay = 0.0001

# Set scheduler parameter
factor = 0.1
patience = 3

# Train
for block in block_list:
    model = ResNet(
        block,
        num_layers
    )
    model.cuda()
    None
    
    # Test
    input = torch.randn(16, 3, 32, 32).cuda()
    model(input).shape
    
    # Set Loss
    loss_fct = nn.CrossEntropyLoss()    

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Set scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

    result = train(epochs)    
    
    save_name = block.__name__
    result_save_path = f"./result/{save_name}.json"
    with open(result_save_path , 'w') as f:
        json.dump(result, f)

    torch.cuda.empty_cache()