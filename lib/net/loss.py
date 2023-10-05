#=========================================
# Written by Yude Wang
#=========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
class BLPLoss(nn.Module):
	def __init__(self, gamma, ignore_index=255):
		super(BLPLoss, self).__init__()
		self.gamma = gamma
		self.ignore_index = ignore_index

	def forward(self, pred1, pred2, scribble_label_onehot, cls_label):
		prob1 = pred1.softmax(dim=1)
		logprob1 = F.log_softmax(pred1, dim=1)

		prob2 = pred2.softmax(dim=1)
		w = scribble_label_onehot[:,:-1] * ((1-prob2[:,:-1]*prob1)**self.gamma).detach()
		loss = - w * logprob1
		loss = torch.sum(loss.flatten(1), dim=-1, keepdim=False)
		num = torch.sum(w.flatten(1),dim=-1,keepdim=False)+1e-7
		loss = torch.mean(loss/num)

		return loss
