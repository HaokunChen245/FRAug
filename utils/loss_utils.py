import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np

class HLoss(nn.Module):
    # larger Entropy, means output distribution is more even, or prediction is not sure.
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1).mean(0)
        return b

def non_saturating_loss(logits, targets):
	probs = logits.softmax(1)
	log_prob = torch.log(1 - probs + 1e-12)
	if targets.ndim == 2:
		return - (targets * log_prob).sum(1).mean()
	else:
		return F.nll_loss(log_prob, targets)

class NonSaturatingLoss(nn.Module):
	def __init__(self, num_classes, epsilon):
		super().__init__()
		self.epsilon = epsilon
		self.num_classes = num_classes

	def forward(self, logits, targets):
		onehot_targets = F.one_hot(targets, self.num_classes).float()
		if self.epsilon > 0: # label smoothing
			targets = (1 - self.epsilon) * onehot_targets + self.epsilon / self.num_classes
		else:
			targets = onehot_targets

		return non_saturating_loss(logits, targets)

class MMD_loss(nn.Module):
	def __init__(self, kernel_type = 'mean_cov', kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.kernel_type = kernel_type
		self.fix_sigma = None
		return
        
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)	

		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2) 
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def mean_cov(self, x, y):
		mean_x = x.mean(0, keepdim=True)
		mean_y = y.mean(0, keepdim=True)
		cent_x = x - mean_x
		cent_y = y - mean_y
		cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
		cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

		mean_diff = (mean_x - mean_y).pow(2).mean()
		cova_diff = (cova_x - cova_y).pow(2).mean()

		return mean_diff + cova_diff

	def forward(self, x, y):
		if 'gaussian' in self.kernel_type:
			batch_size = int(x.size()[0])
			kernels = self.guassian_kernel(x, y, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
			XX = kernels[:batch_size, :batch_size]
			YY = kernels[batch_size:, batch_size:]
			XY = kernels[:batch_size, batch_size:]
			YX = kernels[batch_size:, :batch_size]
			loss = torch.mean(XX + YY - XY -YX)

		else:
			return self.mean_cov(x, y)
        
		return loss
