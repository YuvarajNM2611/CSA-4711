"""Simulate a two-hidden-layer Boltzmann machine with Gibbs updates."""
import numpy as np
rng=np.random.default_rng(10); n_visible,n_h1,n_h2=8,6,4
W1=rng.normal(0,.15,(n_visible,n_h1)); W2=rng.normal(0,.15,(n_h1,n_h2)); vb=np.zeros(n_visible); b1=np.zeros(n_h1); b2=np.zeros(n_h2)
sig=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
v=(rng.random((1,n_visible))<.5).astype(float)
for step in range(1000):
 h1=(rng.random((1,n_h1))<sig(v@W1+b1)).astype(float); h2=(rng.random((1,n_h2))<sig(h1@W2+b2)).astype(float)
 h1=(rng.random((1,n_h1))<sig(v@W1+h2@W2.T+b1)).astype(float); v=(rng.random((1,n_visible))<sig(h1@W1.T+vb)).astype(float)
 if step%200==0: print('step',step,'visible sample',v.astype(int).ravel())
