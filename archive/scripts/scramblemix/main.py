# coding: utf-8
import sys
#import pathlib:wq

sys.path.append('../../models')
sys.path.append('../../utils')
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
#from models.no_adaptation_network import ShakePyramidNet
from no_adaptation_network import ShakePyramidNet
import tensorboardX as tbx
from scheduler import CyclicLR
import random
from torch.optim import lr_scheduler
import os
from parameter import get_parameters
from Blockwise_scramble_LE import blockwise_scramble
from dataloader import training_dataloader, test_dataloader
from trainer import Trainer 
from wideresnet import WideResNet
from densenet import  DenseNet3
from vgg import vgg19_bn
from resnext import senet
from resnext2 import ResNeXt29_16x4d
if __name__ == "__main__":
  seed = 130
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  args = get_parameters() 
  trainloader = training_dataloader(dataset_name=args.dataset)
  testloader = test_dataloader(dataset_name=args.dataset)

  permutation = [i for i in range(args.num_of_keys)]

  if args.model_name == "densenet":
    if args.dataset == "cifar10":
      net = DenseNet3(num_classes=10)
    elif args.dataset == "svhn":
      net = DenseNet3(num_classes=10)
    elif args.dataset == "cifar100":
      net = DenseNet3(num_classes=100)
  elif args.model_name == "wideresnet":
    if args.dataset == "cifar10":
      net = WideResNet(num_classes=10)
    elif args.dataset == "svhn":
      net = WideResNet(num_classes=10)
    elif args.dataset == "cifar100":
      net = WideResNet(num_classes=100)
  elif args.model_name == "senet":
    print("senet")
    if args.dataset == "cifar10":
      net = senet(num_classes=10)
    elif args.dataset == "svhn":
      net = senet(num_classes=10)
    elif args.dataset == "cifar100":
      net = senet(num_classes=100)
  elif args.model_name == "senet2":
    print("senet2")
    if args.dataset == "cifar10":
      net = ResNeXt29_16x4d(num_classes=10)
    elif args.dataset == "svhn":
      net = ResNeXt29_16x4d(num_classes=10)
    elif args.dataset == "cifar100":
      net = ResNeXt29_16x4d(num_classes=100)
  else:
    if args.dataset == "cifar10":
      net = ShakePyramidNet(depth=args.depth, alpha=args.alpha, label=10)
    elif args.dataset == "svhn":
      net = ShakePyramidNet(depth=args.depth, alpha=args.alpha, label=10)
    elif args.dataset == "cifar100":
      net = ShakePyramidNet(depth=args.depth, alpha=args.alpha, label=100)

  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(),
                lr=args.lr,
                momentum=args.momentum, 
                weight_decay=args.weight_decay)
  writer = tbx.SummaryWriter(args.tensorboard_name)
  scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')])
  if not os.path.isdir(args.save_directory_name):
      os.makedirs(args.save_directory_name)

  model_trainer = Trainer(trainloader, testloader, net, criterion, optimizer, blockwise_scramble, args)

  for epoch in range(args.e):
    scheduler.step()
    train_loss, train_acc = model_trainer.train()
    if epoch == args.e - 1:
      for i in range(10):
        test_loss, test_acc, test_acc_tta4 = model_trainer.test(1)
        print(train_acc, test_acc, test_acc_tta4)
    else:
      test_loss, test_acc, test_acc_tta4 = model_trainer.test(0)
      # writer.add_scalars('data/loss',
      # {
      #   'train_loss': train_loss, 'test_loss': test_loss,
      # },
      #   (epoch + 1)
      # )
      # writer.add_scalars('data/acc',
      # {
      #   'train_acc': train_acc, 'test_acc': test_acc
      # },
      #   (epoch + 1)
      # )
      print(train_acc, test_acc, test_acc_tta4)
  print("best acc : ",model_trainer.best_acc)
  writer.export_scalars_to_json("./"+args.json_file_name)
  writer.close()
