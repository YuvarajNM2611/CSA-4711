"""Backpropagate through a tanh RNN and inspect gradients over time."""
import numpy as np
rng=np.random.default_rng(2); T=20; x=rng.normal(size=T); Wh=.9; Wx=.5; h=np.zeros(T+1)
for t in range(T): h[t+1]=np.tanh(Wh*h[t]+Wx*x[t])
# gradient of final state w.r.t. each earlier hidden state input
for lag in (1,5,10,19):
 grad=1.
 for t in range(T-lag,T): grad*=Wh*(1-h[t+1]**2)
 print(f'lag {lag:2}: |d h_T / d h_(T-lag)| = {abs(grad):.6g}')
