from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import *
import torch.optim as optim
from torchvision import transforms
import torchvision
import torch.nn.functional as F

class LinfPGDAttack:
	def __init__(self, model, epsilon, k, a, random_start):
		self.model = model
		self.epsilon = epsilon
		self.k = k
		self.a = a
		self.rand = random_start
		self.criterion = nn.CrossEntropyLoss()

	def perturb(self, x_nat, y):
		if self.rand:
			x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
		else:
			x = np.copy(x_nat)

		for i in range(self.k):
			x_var = Variable(torch.FloatTensor(x), requires_grad=True)
			y_true = Variable(torch.LongTensor(y), requires_grad=False)
			outputs = self.model(x_var)
			loss = self.criterion(outputs, y_true)
			loss.backward() 
			grad = x_var.grad.data
			x += self.a * np.sign(grad)

			x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
			x = np.clip(x, 0, 1)

		return x

class FGSAttack:
	def __init__(self, model, epsilon):
		self.model = model
		self.epsilon = epsilon
		self.criterion = nn.CrossEntropyLoss()

	def perturb(self, x_nat, y):
		
		x = np.copy(x_nat)

		x_var = Variable(torch.FloatTensor(x), requires_grad=True)
		y_true = Variable(torch.LongTensor(y), requires_grad=False)
		outputs = self.model(x_var)
		loss = self.criterion(outputs, y_true)
		loss.backward() 
		grad = x_var.grad.data
		x += self.epsilon * np.sign(grad)
		x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
		x = np.clip(x, 0, 1)

		return x

