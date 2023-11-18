from learnable_encryption_augmix import BlockScramble
import numpy as np
from PIL import Image
import torch

# def blockwise_scramble(imgs,idx):
#   x_stack  = None
#   imgs = np.asarray(imgs) / 255.0
#   #exit()
#   key_file = 'key4/'+str(idx)+'_.pkl'
#   bs = BlockScramble( key_file )
#   imgs = bs.Scramble(imgs.reshape([1, 32, 32, 3])).reshape([32, 32, 3])
#   x_stack = np.uint8(imgs * 255).reshape(32,32,3)
#   x_stack = Image.fromarray(x_stack)
#   return x_stack

def blockwise_scramble(imgs,idx):
  x_stack  = None
  imgs = imgs.reshape([1,3,32,32])
  key_file = 'key4/'+str(idx)+'_.pkl'
  bs = BlockScramble( key_file )
  out = np.transpose(imgs,(0, 2, 3, 1))
  x_stack = bs.Scramble(out.reshape([out.shape[0],32,32,3])).reshape([out.shape[0],32,32,3])
  x_stack = np.transpose(x_stack,(0,3,1,2))
  return torch.from_numpy(x_stack).squeeze()
