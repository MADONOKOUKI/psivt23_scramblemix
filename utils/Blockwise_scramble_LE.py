from learnable_encryption import BlockScramble
import numpy as np

def blockwise_scramble(imgs,idx):
  x_stack  = None
  key_file = '/groups1/gaa50073/madono/icip2021_classification/utils/key4/'+str(idx)+'_.pkl'
  bs = BlockScramble( key_file )
  imgs = bs.Scramble(np.transpose(imgs,(0, 2, 3, 1)))
  return np.transpose(np.transpose(imgs,(0, 3, 1, 2)))
#  for k in range(8):
#    tmp = None
#    for j in range(8):
#      key_file = '/groups1/gaa50073/madono/icip2021_classification/utils/key4/'+str(idx)+'_.pkl'
#      bs = BlockScramble( key_file )
#      out = np.transpose(imgs,(0, 2, 3, 1))[:,k*4:(k+1)*4,j*4:(j+1)*4,:]
  #    out = out[:,k*4:(k+1)*4,j*4:(j+1)*4,:]
 #     out = bs.Scramble(out).reshape([imgs.shape[0],4,4,3])
  #    if tmp is None:
   #     tmp = out
#      else:
#        tmp = np.concatenate((tmp,out),axis=2)
#    if x_stack is None:
#      x_stack = tmp
#    else:
#      x_stack = np.concatenate((x_stack,tmp),axis=1)
#  x_stack = np.transpose(x_stack,(0,3,1,2))
 # return x_stack

def blockwise_decramble(imgs,idx):
  x_stack  = None
  for k in range(8):
    tmp = None
   # x_stack = None
    for j in range(8):
      key_file = 'key4/'+str(idx)+'_.pkl'
      bs = BlockScramble( key_file )
      out = np.transpose(imgs,(0, 2, 3, 1))
      out = out[:,k*4:(k+1)*4,j*4:(j+1)*4,:]
      out = bs.Decramble(out.reshape([out.shape[0],4,4,3])).reshape([out.shape[0],4,4,3])
      if tmp is None:
        tmp = out
      else:
        tmp = np.concatenate((tmp,out),axis=2)
    if x_stack is None:
      x_stack = tmp
    else:
      x_stack = np.concatenate((x_stack,tmp),axis=1)
  x_stack = np.transpose(x_stack,(0,3,1,2))
  return x_stack

