from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
from timeit import default_timer as timer
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import *
import torch.optim as optim
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from pgd_attack import LinfPGDAttack, FGSAttack

class CNN(nn.Module):
	def __init__(self, n_classes):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 2, 5, 1)
		self.conv2 = nn.Conv2d(2, 4, 5, 1)
		self.fc1 = nn.Linear(64, 10)
		
	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(x, 2, 2)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*4)
		x = self.fc1(x)
		return x


with open('config.json') as config_file:
	config = json.load(config_file)

# Setting up training parameters
torch.manual_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5,), (1.0,))])
traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)

net = CNN(10)
natural_net = CNN(10)

# Setting up the optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)
natural_optimizer = optim.Adam(natural_net.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
k_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

model_dir = 'models/cat_train'#config['model_dir']
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

shutil.copy('config.json', model_dir)
training_time = 0.0

for k in k_set:
	total_iter = 100
	epoch = 0
	attack = LinfPGDAttack(net, 
					   config['epsilon'],
					   k,
					   config['a'],
					   config['random_start'])
	while total_iter > 0 and epoch < max_num_training_steps:
		running_loss = 0.0 
		print("Epoch: {} | k = {}".format(epoch, k))
		for data in tqdm(trainloader):
			total_iter -= 1

			inputs, labels = data 
			inputs, labels = Variable(inputs), Variable(labels)
			optimizer.zero_grad() 
	
			# Compute Adversarial Perturbations
			start = timer()
			x_adv = attack.perturb(inputs.data.numpy(), labels.data.numpy())
			x_adv_v = Variable(torch.FloatTensor(x_adv))
			end = timer()
			training_time += end - start

			outputs = net(x_adv_v)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]

			natural_outputs = natural_net(inputs)
			loss = criterion(natural_outputs, labels)
			loss.backward()
			natural_optimizer.step()

		print('Epoch: {} | Loss: {}'.format(epoch, running_loss/2000.0))
		correct = 0.0 
		natural_correct = 0.0 
		total = 0 
		for data in testloader:
			images, labels = data
			outputs = net(Variable(images))
			natural_outputs = natural_net(Variable(images))
			_, predicted = torch.max(outputs.data, 1)
			_, natural_pred = torch.max(natural_outputs.data, 1)
			total += labels.size(0) 
			correct += (predicted == labels).sum()
			natural_correct += (natural_pred == labels).sum()

		print("Adv Accuracy: {}".format(correct/total))
		print("Nat Accuracy: {}".format(natural_correct/total))
		torch.save(net.state_dict(), '{0}/models_epoch_{1}.pth'.format(model_dir, epoch))
		epoch += 1
	



