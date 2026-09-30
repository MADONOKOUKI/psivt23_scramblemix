import torch
import random
import numpy as np
from copy import deepcopy

# np.random.seed()
def negaposi(val):
  # np.random.seed()
  p = np.random.randint(0,2)
  if p == 0:
    val = 255 - val
  return val

def channel_change(val):
  p = [0, 1, 2]
  # np.random.seed()
  random.shuffle(p)
  val[0], val[1], val[2] = val[p[0]], val[p[1]], val[p[2]]
  return val

perm = [ [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]

def pixel_based_encryption(img, pixel_reverse, channel_shuffle):
    img = np.array(img) #/ 255.0
    img = img * pixel_reverse.reshape([32,32,3]) + (255 - img) * (1 - pixel_reverse.reshape([32,32,3]))
    imgs = []
    for i in range(6):
      img0 = deepcopy(img)
      img0[:,:,0], img0[:,:,1], img0[:,:,2] = img0[:,:,perm[i][0]], img0[:,:,perm[i][1]], img0[:,:,perm[i][2]]
      imgs.append(img0)
    val = np.zeros(img.shape)
    for i in range(6):
      for j in range(3):
        val[:,:,j] = val[:,:,j] + deepcopy(imgs[i][:,:,j]) * (channel_shuffle == i).reshape(32,32)
    return val


