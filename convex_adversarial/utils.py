#coding=utf-8
import math
import numpy as np
import pylab as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class VariableSizeInspector(nn.Module):
    def __init__(self):
        super(VariableSizeInspector, self).__init__()
    def forward(self, x):
        #print x.data[0, 0], x.data[0, 1], x.data[0,2]
        return x
    def __repr__(self):
        return 'VariableSizeInspector()'


def drawGraph(x,y):
  pl.title("The Convex Hull")
  pl.xlabel("x axis")
  pl.ylabel("y axis")
  pl.plot(x,y,'ro', markersize=2)
def drawCH(x,y):
  #pl.plot(x,y1,label='DP',linewidth=4,color='red')
  pl.plot(x,y,color='blue',linewidth=2)
  pl.plot(x[-1],y[-1],x[0],y[0])
  lastx=[x[-1],x[0]]
  lasty=[y[-1],y[0]]
  pl.plot(lastx,lasty,color='blue',linewidth=2)
def matrix(rows,cols):
  cols=3
  mat = [[0 for col in range (cols)]
        for row in range(rows)]
  return mat
def crossMut(stack,p3):
  p2=stack[-1]
  p1=stack[-2]
  vx,vy=(p2[0]-p1[0],p2[1]-p1[1])
  wx,wy=(p3[0]-p1[0],p3[1]-p1[1])
  return (vx*wy-vy*wx)
def GrahamScan(mat):
  #print mat
  points=len(mat) #点数
  """
 for k in range(points):
  print mat[k][0],mat[k][1],mat[k][2]
 """
  stack=[]
  stack.append((mat[0][0],mat[0][1])) #push p0
  stack.append((mat[1][0],mat[1][1])) #push p1
  stack.append((mat[2][0],mat[2][1])) #push p2
  for i in range(3,points):
    #print stack
    p3=(mat[i][0],mat[i][1])
    while crossMut(stack,p3)<0:stack.pop()
    stack.append(p3)
  return stack
def drawGraphandOuter(pool, color='b'):
    outerx = []
    outery = []
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

    plt.plot(stack[:,0], stack[:,1], c=color)
