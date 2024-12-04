import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

from tqdm import tqdm
import wandb

import random
import datetime
import itertools

from train import train_epoch
from val import val_epoch
from ResNet import ResNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.mps:
    device = "mps"

print(f"device:{device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomResizedCrop((224, 224)),  # 随机裁剪并调整到224x224
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.Flowers102(root="../data", split="train", download=True, transform=transform)
val_dataset = datasets.Flowers102(root="../data", split="val", download=True, transform=transform)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)


def main():
    lr_list = [1e-3]
    num_epochs_list = [300]
    momentum_list = [0.85, 0.9, 0.95]
    batch_size_list = [256]
    seed_list = [42]
    weight_decay_list = [1e-2, 1e-3, 1e-4]
    
    for lr, num_epochs, momentum, batch_size, seed, weight_decay in itertools.product(lr_list, num_epochs_list, momentum_list, batch_size_list, seed_list, weight_decay_list):
        
        print(f"lr:{lr}, num_epochs:{num_epochs}, momentum:{momentum}, batch_size:{batch_size}, seed:{seed}")
        
        wandb.init(
            project="ResNet - Flower102",
            name=f"momentum:{momentum} weight decay:{weight_decay} lr:{lr}",
            config={
                "learning rate": lr,
                "num_epochs": num_epochs,
                "momentum": momentum,
                "batch_size": batch_size,
                "seed": seed,
                "weight decay": 1e-4
            }
        )
        
        set_seed(seed)
        
        model = ResNet(in_channel=3, out_channel=102).to(device)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(model, train_loader, optimizer, criterion, epoch+1, num_epochs, device)
            # wandb.log({
            #     'train_loss':train_loss,
            #     'train_acc':train_correct/len(train_dataset)
            # }, step=epoch) 
            
            val_loss, val_correct = val_epoch(model, val_loader, optimizer, criterion, epoch+1, num_epochs, device)
        #     wandb.log({
        #         'val_loss': val_loss,
        #         'val_acc': val_correct/len(val_dataset)
        #     }, step=epoch)
        
        # wandb.finish()
        

if __name__ == '__main__':
    main()
    pass
