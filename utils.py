#coding=utf-8
import math
import numpy as np
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

def drawGraphandOuter(pool, color='b'):
    outerx = []
    outery = []
    for p in pool:
        xx, yy = p
        outerx.append(xx)
        outery.append(yy)
    plt.scatter(outerx, outery, cmap="coolwarm", s=1)    
    stack = convex_hull(pool)
    stack.append(stack[0])
    stack = np.asarray(stack)

    plt.plot(stack[:,0], stack[:,1], c=color)
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]
