#coding=utf-8
import math
import numpy
import pylab as pl
#画原始图
def drawGraph(x,y):
  pl.title("The Convex Hull")
  pl.xlabel("x axis")
  pl.ylabel("y axis")
  pl.plot(x,y,'ro', markersize=2)
#画凸包
def drawCH(x,y):
  #pl.plot(x,y1,label='DP',linewidth=4,color='red')
  pl.plot(x,y,color='blue',linewidth=2)
  pl.plot(x[-1],y[-1],x[0],y[0])
  lastx=[x[-1],x[0]]
  lasty=[y[-1],y[0]]
  pl.plot(lastx,lasty,color='blue',linewidth=2)
#存点的矩阵,每行一个点,列0->x坐标,列1->y坐标,列2->代表极角
def matrix(rows,cols):
  cols=3
  mat = [[0 for col in range (cols)]
        for row in range(rows)]
  return mat
#返回叉积
def crossMut(stack,p3):
  p2=stack[-1]
  p1=stack[-2]
  vx,vy=(p2[0]-p1[0],p2[1]-p1[1])
  wx,wy=(p3[0]-p1[0],p3[1]-p1[1])
  return (vx*wy-vy*wx)
#Graham扫描法O(nlogn),mat是经过极角排序的点集
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
