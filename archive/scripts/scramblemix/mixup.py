#https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
import numpy as np
import torch
def mixup_data(x1, x2,  y, alpha=0.2, use_cuda=True):

    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x1 + (1 - lam) * x2
    y_a, y_b = y, y

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
