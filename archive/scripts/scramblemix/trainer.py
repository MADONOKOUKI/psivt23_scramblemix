import torch
import torch.nn as nn
import numpy as np
from mixup import *
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import lpips
from Blockwise_scramble_LE import blockwise_scramble
from copy import deepcopy
seed = 130
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from parameter import get_parameters
args = get_parameters() 
class Trainer(object):
    def __init__(self, trainloader, testloader, net, criterion, optimizer, blockwise_scramble, args):

      self.trainloader = trainloader
      self.testloader = testloader
      self.net = net
      self.criterion = criterion
      self.optimizer = optimizer
      self.scramble = blockwise_scramble
      self.save_directory_name = args.save_directory_name
      self.best_acc = 0
      self.alpha = 0.2 
      self.iqa = lpips.LPIPS(net='alex').cuda()
      self.args = args

    def train(self):

      self.net.train()
      total_loss = 0
      correct = 0
      total = 0
      for batch_idx, (inputs, inputs1, inputs2, inputs3, inputs4, inputs5, targets) in enumerate(self.trainloader):

        inputs1, inputs2, inputs3, inputs4, targets = inputs1.cuda(), inputs2.cuda(), inputs3.cuda(), inputs4.cuda(),  targets.cuda()

        self.optimizer.zero_grad()

        outputs_list = []
        
        # assumed max STR is 3
        outputs1, _ = self.net(inputs1)
        outputs2, _ = self.net(inputs2)
        outputs3, _ = self.net(inputs3)
        outputs4, _ = self.net(inputs4)

        outputs_list.append(outputs1)
        outputs_list.append(outputs2)
        outputs_list.append(outputs3)
        outputs_list.append(outputs4)

        p_mixture = 0
        for i in range(self.args.num_of_TTA):
          p_mixture = p_mixture + F.softmax(outputs_list[i], dim=1)

        # p_mixture = torch.clamp(p_mixture / args.num_of_TTA, 1e-7, 1).log()
        p_mixture = (p_mixture / self.args.num_of_TTA).log()
        loss_js =0
        for i in range(self.args.num_of_TTA):
          loss_js = loss_js + F.kl_div(p_mixture, F.softmax(outputs_list[i], dim=1), reduction='batchmean')
        loss_js = loss_js / self.args.num_of_TTA

        loss = 0
        for i in range(self.args.num_of_TTA):
          # print(inputs1.size())
          # print(outputs_list[i].size(), targets.size())
          loss = loss + self.criterion(outputs_list[i], targets)
        loss = loss / self.args.num_of_TTA

        if self.args.js_divergence_regularization:
          loss = loss + loss_js

        loss.backward()
        self.optimizer.step()

        for i in range(self.args.num_of_TTA):
          total_loss += loss.item()
          _, predicted = outputs_list[i].max(1)
          total += targets.size(0)
          correct += predicted.eq(targets.data).cpu().sum().float().item()

      return total_loss/total, 100.*correct/total

    def test(self, switch):

      self.net.eval()
      total_loss = 0
      correct = 0
      total = 0
      total_lpips_loss = 0
      correct_tta1 = 0
      total_tta1 = 0
      correct_tta4 = 0
      total_tta4 = 0      
      with torch.no_grad():
        for batch_idx, (inputs, scrambled_inputs, mix1, mix2, mix3, mix4, targets) in enumerate(self.testloader):
          total_outputs = 0
          scrambled = None
          loop = 1
          inputs, scrambled_inputs, targets = inputs.cuda(), scrambled_inputs.cuda(), targets.cuda()
          mix1, mix2, mix3, mix4 = mix1.cuda(), mix2.cuda(), mix3.cuda(), mix4.cuda()
          mixes = [mix1, mix2, mix3, mix4]

          # random sample inference
          outputs, _ = self.net(scrambled_inputs)
          loss = self.criterion(outputs, targets)
          _, predicted = (outputs).max(1)
          total += targets.size(0)
          correct = correct + predicted.eq(targets.data).cpu().sum().float().item()

          # tta
          for idx in range(4):
            outputs_tta, _ = self.net(mixes[idx])
            total_outputs = total_outputs + outputs_tta
          total_outputs = total_outputs / 4.0
          _, predicted = (total_outputs).max(1)
          correct_tta4 = correct_tta4 + predicted.eq(targets.data).cpu().sum().float().item()

          if switch == 1:
            if batch_idx==0:
              save_image(inputs[0:64].data, str(self.save_directory_name) + "/inputs_lpips_lists.png", nrow=8, normalize=True)
              save_image(scrambled_inputs[0:64].data, str(self.save_directory_name) +"/answer_lpips_lists.png", nrow=8, normalize=True)
              for j in range(64):
                print(str(j)+"'s lpips score = ", self.iqa.forward(inputs[j].view(1,3,32,32), scrambled_inputs[j].view(1,3,32,32)).data)
                print(F.softmax(outputs[j]))
                print(F.softmax(total_outputs[j]))
                save_image(inputs[j].data, str(self.save_directory_name) + "/inputs_%d.png" % j, nrow=1, normalize=True)
                save_image(scrambled_inputs[j].data, str(self.save_directory_name) + "/scrambled_inputs_%d.png" % j, nrow=1, normalize=True)
            for j in range(scrambled_inputs.size()[0]):
              total_lpips_loss = total_lpips_loss + self.iqa.forward(inputs[j].view(1,3,32,32) , scrambled_inputs[j].view(1,3,32,32)).sum().cpu().data
        if switch == 1:
          print(total_lpips_loss.item() / 10000)  
      # Save checkpoint.
      acc = 100.*correct/total
      if self.best_acc < acc:
            self.best_acc = acc
            torch.save(self.net.state_dict(), "model_" + args.save_directory_name)
      return total_loss/total, 100.*correct/total,  100.*correct_tta4/total

