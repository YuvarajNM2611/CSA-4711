"""Train a perceptron on AND and show a two-layer forward pass."""
import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([0,0,0,1])
w=np.zeros(2); b=0.; lr=.1
for _ in range(20):
    for x,t in zip(X,y):
        pred=int(w@x+b>=0); err=t-pred; w+=lr*err*x; b+=lr*err
print('AND predictions:',(X@w+b>=0).astype(int),'weights:',w,'bias:',b)
# one hidden layer forward propagation with sigmoid activation
sig=lambda z: 1/(1+np.exp(-z))
W1=np.array([[.5,-.4],[.3,.8]]); b1=np.zeros(2); W2=np.array([[.7],[-.6]]); b2=np.zeros(1)
h=sig(X@W1+b1); out=sig(h@W2+b2)
print('Two-layer network probabilities:',out.ravel())
