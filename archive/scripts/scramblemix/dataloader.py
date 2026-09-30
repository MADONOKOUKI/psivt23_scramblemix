import torch
import torchvision
import torchvision.transforms as transforms
from cifar10 import AugMixDataset

def training_dataloader(train=True, download=True, batch_size=512, shuffle=True, num_workers=16,dataset_name="cifar10"):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    preprocess = transforms.Compose([
       transforms.ToTensor(),
    ])
 
    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='.././data_cifar10', train=True, download=True, transform=transform_train)
    elif dataset_name == "svhn":
        trainset = torchvision.datasets.SVHN(root='.././data_svhn', split='train', download=True, transform=transform_train)
    elif dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='.././data_cifar100', train=True, download=True, transform=transform_train)
 
    trainset = AugMixDataset(trainset, preprocess, True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

    return trainloader

def test_dataloader(train=False, download=True, batch_size=128, shuffle=False, num_workers=16, dataset_name="cifar10"):
    
    preprocess = transforms.Compose([
       transforms.ToTensor(),
    ])

    if dataset_name == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='.././data_cifar10', train=False, download=True, transform=None)
    elif dataset_name == "svhn":
        testset = torchvision.datasets.SVHN(root='.././data_svhn', split='test', download=True, transform=None)
    elif dataset_name == "cifar100":
        testset = torchvision.datasets.CIFAR100(root='.././data_cifar100', train=False, download=True, transform=None)

    testset = AugMixDataset(testset, preprocess, False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    return testloader
