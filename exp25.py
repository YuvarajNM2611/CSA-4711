"""Fit an RBM to binarized handwritten digits and inspect reconstruction."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
X,y=load_digits(return_X_y=True); X=(X/16.>.5).astype(float)
rbm=BernoulliRBM(n_components=64,learning_rate=.05,batch_size=32,n_iter=20,random_state=2).fit(X)
H=rbm.transform(X); recon=rbm.gibbs(X[:100]);
print('Hidden representation shape:',H.shape); print('Mean reconstruction error:',np.mean((X[:100]-recon)**2)); print('First label:',y[0])
