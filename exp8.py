"""Train a small two-layer network on XOR using NumPy backpropagation."""
import numpy as np
rng=np.random.default_rng(1); X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([[0],[1],[1],[0]],float)
sig=lambda z:1/(1+np.exp(-z))
W1=rng.normal(0,.7,(2,8)); b1=np.zeros((1,8)); W2=rng.normal(0,.7,(8,1)); b2=np.zeros((1,1)); lr=1.
for epoch in range(10000):
    h=sig(X@W1+b1); p=sig(h@W2+b2); d2=p-y; d1=(d2@W2.T)*h*(1-h)
    W2-=lr*h.T@d2/len(X); b2-=lr*d2.mean(0,keepdims=True); W1-=lr*X.T@d1/len(X); b1-=lr*d1.mean(0,keepdims=True)
print('XOR probabilities:',sig(sig(X@W1+b1)@W2+b2).ravel().round(3))
