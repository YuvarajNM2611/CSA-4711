"""Implement a binary RBM and compare CD-k reconstruction errors."""
import numpy as np
rng=np.random.default_rng(3); X=(rng.random((300,12))<.35).astype(float)
W=rng.normal(0,.1,(12,6)); vb=np.zeros(12); hb=np.zeros(6)
sig=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
for k in (1,3,5):
 w=W.copy(); v0=X.copy()
 for _ in range(30):
  ph=sig(v0@w+hb); h=(rng.random(ph.shape)<ph).astype(float); vk=v0
  for __ in range(k):
   pv=sig(h@w.T+vb); vk=(rng.random(pv.shape)<pv).astype(float); h=(rng.random(ph.shape)<sig(vk@w+hb)).astype(float)
  w+=.03*(v0.T@ph-vk.T@sig(vk@w+hb))/len(X)
  vb+=.03*np.mean(v0-vk,axis=0); hb+=.03*np.mean(ph-sig(vk@w+hb),axis=0)
 recon=sig(sig(X@w+hb)@w.T+vb); print(f'CD-{k} reconstruction MSE:',np.mean((X-recon)**2))
