import torch
import torch.nn as nn
from torch.autograd import Variable
from convex_adversarial import DualNetBounds
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import random
np.random.seed(1)
torch.manual_seed(231)
epsilon = 0.5
h = 3
fixed = True

def draw_adv(net, X):
	output = net(Variable(X))
	OX = X.clone()
	k=30
	z_hat = []
	for i in range(-k, k+1):
		dx = float(i) / k * epsilon
		for j in range(-k, k+1):
			dy = float(j) / k * epsilon
			X[:, 0] = OX[:, 0] + dx
			X[:, 1] = OX[:, 1] + dy
			z_hat.append(net(Variable(X)).data.numpy())
		
	z_hat = np.vstack(z_hat)    
	plt.plot(output.data.numpy()[:,0], output.data.numpy()[:,1], 'ro')
	plt.scatter(z_hat[:,0], z_hat[:,1], cmap="coolwarm", s=10)

def simplex(net):
	x1 = 0.417022
	x2 = 0.72032449
	params = [p for p in net.parameters()]
	a = params[0].data.numpy()
	b = params[1].data.numpy()
	d = params[2].data.numpy()
	e = params[3].data.numpy()
	vname = ['z11','z12', 'z^21','z^22','z^23', 'z21', 'z22', 'z23','z^31','z^32']
	
	lts = [ [0,0,-zu[0],0,0,zu[0]-zl[0],0,0,0,0],[0,0,1,0,0,-1,0,0,0,0],
			[0,0,0,-zu[1],0,0,zu[1]-zl[1],0,0,0],[0,0,0,1,0,0,-1,0,0,0],
			[0,0,0,0,-zu[2],0,0,zu[2]-zl[2],0,0],[0,0,0,0,1,0,0,-1,0,0]]
	ltB = [-zu[0]*zl[0],0,-zu[1]*zl[1],0,-zu[2]*zl[2],0]
	
	eqs = [[a[0,0],a[0,1],-1,0,0,0,0,0,0,0],[a[1,0],a[1,1],0,-1,0,0,0,0,0,0],[a[2,0],a[2,1],0,0,-1,0,0,0,0,0],
       	[0,0,0,0,0,-d[0,0],-d[0,1],-d[0,2],1,0],
       	[0,0,0,0,0,-d[1,0],-d[1,1],-d[1,2],0,1]]
	eqB = [-b[0], -b[1], -b[2],e[0], e[1]]

	z11_bnd = (-(x1+epsilon),x1+epsilon)
	z12_bnd = (-(x2+epsilon),x2+epsilon)
	z2_bnd = (0, None)
	ubnd = (None, None)
	outerx = []
	outery = []
	pool = []
	c = [0,0,0,0,0,0,0,0,1,1]
	res = linprog(c, lts, ltB, eqs, eqB, 
			bounds=(z11_bnd, z12_bnd, ubnd, ubnd, ubnd, z2_bnd, z2_bnd, z2_bnd, ubnd, ubnd),
			options={'disp': True, 'bland': False, 'tol': 1e-12, 'maxiter': 1000})
	print res
	
	for i in range(0,1000):
		k = i / 100
		c =  [0]*k + random.sample(xrange(-100, 100), 10-k)
		res = linprog(c, lts, ltB, eqs, eqB, 
			bounds=(z11_bnd, z12_bnd, ubnd, ubnd, ubnd, z2_bnd, z2_bnd, z2_bnd, ubnd, ubnd),
			options={'disp': False, 'bland': False, 'tol': 1e-12, 'maxiter': 1000})
		pool.append(tuple(res.x[-2:]))
	pool = set(pool)
	for p in pool:
		xx, yy = p
		outerx.append(xx)
		outery.append(yy)

	plt.scatter(outerx, outery, cmap="coolwarm", s=10)
	drawGraph(outerx, outery)
	#plt.scatter(outerx, outery, color='r')
	max_num = len(outerx)
	mat = matrix(max_num, 3) 
	minn = 300
	for i in range(max_num):
		mat[i][0], mat[i][1] = outerx[i], outery[i]
		if outery[i] < minn: 
			minn = outery[i]
			tmp = i
	d = {}  
	for i in range(max_num):    
		if (mat[i][0],mat[i][1]) == (outerx[tmp],outery[tmp]) : mat[i][2]=0
		else: mat[i][2] = math.atan2((mat[i][1] - outery[tmp]), (mat[i][0] - outerx[tmp]))
		d[(mat[i][0],mat[i][1])] = mat[i][2]
	lst = sorted(d.items(), key=lambda e : e[1]) 
	for i in range(max_num):   
		((outerx,outery),eth0) = lst[i]
		mat[i][0], mat[i][1],mat[i][2] = outerx, outery, eth0
	stack = GrahamScan(mat)
	stack.append(stack[0])
	stack = np.asarray(stack)

	plt.plot(stack[:,0], stack[:,1], c='r')
	
def fixed_y(net):
	k = 30
	delta = yu1-yl1
	for i in range(0, k+1):
		dy = float(i) / k * delta
		y1 = yl1+dy
		print y1
		'''
		x1 = 0.417022
		x2 = 0.72032449
		params = [p for p in net.parameters()]
		a = params[0].data.numpy()
		b = params[1].data.numpy()
		d = params[2].data.numpy()
		e = params[3].data.numpy()
		vname = ['z11','z12', 'z^21','z^22','z^23', 'z21', 'z22', 'z23']

		lts = [ [0,0,-zu[0],0,0,zu[0]-zl[0],0,0],[0,0,1,0,0,-1,0,0],
				[0,0,0,-zu[1],0,0,zu[1]-zl[1],0],[0,0,0,1,0,0,-1,0],
				[0,0,0,0,-zu[2],0,0,zu[2]-zl[2]],[0,0,0,0,1,0,0,-1]]
		ltB = [-zu[0]*zl[0],0,-zu[1]*zl[1],0,-zu[2]*zl[2],0]
	
		eqs = [[a[0,0],a[0,1],-1,0,0,0,0,0],[a[1,0],a[1,1],0,-1,0,0,0,0],[a[2,0],a[2,1],0,0,-1,0,0,0],
       		[0,0,0,0,0,-d[0,0],-d[0,1],-d[0,2]]]
		eqB = [-b[0], -b[1], -b[2],e[0]-y1]
		c_min = [0,0,0,0,0,-d[1,0],-d[1,1],-d[1,2]]
		c_max = [0,0,0,0,0,d[1,0],d[1,1],d[1,2]]

		z11_bnd = (-(x1+epsilon), x1+epsilon)
		z12_bnd = (-(x2+epsilon), x2+epsilon)
		z23_bnd = (0, None)
		ubnd = (None, None)
		outerx = []
		outery = []
		pool = []
		res_min = linprog(c_min, lts, ltB, eqs, eqB, 
			bounds=(z11_bnd, z12_bnd, ubnd, ubnd, ubnd, z23_bnd, z23_bnd, z23_bnd),
			options={'disp': False, 'bland': False, 'tol': 1e-12, 'maxiter': 1000})
		res_max = linprog(c_max, lts, ltB, eqs, eqB, 
			bounds=(z11_bnd, z12_bnd, ubnd, ubnd, ubnd, z23_bnd, z23_bnd, z23_bnd),
			options={'disp': False, 'bland': False, 'tol': 1e-12, 'maxiter': 1000})
		
		if res_min.status == 0: 
			plt.plot(y1, res_min.fun+e[1], 'go')
			#plt.plot(y1, -res_max.fun+e[1], 'go')
		'''
net = nn.Sequential(
	nn.Linear(2,h),
	#VariableSizeInspector(),
	nn.ReLU(),
	nn.Linear(h,2)
 )

x = [(0.417022  ,  0.72032449)]
X = torch.Tensor(np.array(x))
#if fixed: net.load_state_dict(torch.load('random_net_0.pth'))

dual = DualNetBounds(net, Variable(X), epsilon)
zu = dual.zu[0].data.numpy()#[0]
zl = dual.zl[0].data.numpy()#[0]
yu1 = dual.zu[1].data.numpy()[1]
yl1 = dual.zl[1].data.numpy()[1]
print yu1
print yl1
draw_adv(net, X)
simplex(net)
#fixed_y(net)
plt.show()
