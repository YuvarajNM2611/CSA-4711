"""Optimize a quadratic with batch GD, momentum, and Adam."""
import numpy as np

def f(x): return np.sum((x-np.array([3.,-2.]))**2)
def optimize(kind, steps=100):
    x=np.array([-5.,5.]); v=np.zeros_like(x); m=np.zeros_like(x); q=np.zeros_like(x); lr=.08
    for t in range(1,steps+1):
        g=2*(x-np.array([3.,-2.]))
        if kind=='SGD': x-=lr*g
        elif kind=='Momentum': v=.9*v+g; x-=lr*v
        else:
            m=.9*m+.1*g; q=.999*q+.001*g*g
            x-=lr*(m/(1-.9**t))/(np.sqrt(q/(1-.999**t))+1e-8)
    return x,f(x)
for name in ('SGD','Momentum','Adam'): print(name, optimize(name))
