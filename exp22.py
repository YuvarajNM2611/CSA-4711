"""Separate mixed sources with FastICA and plot original/recovered signals."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
rng=np.random.default_rng(1); t=np.linspace(0,8,2000)
S=np.c_[np.sin(2*t),np.sign(np.sin(3*t))]+.15*rng.normal(size=(len(t),2)); A=np.array([[1,1],[.5,2.]])
X=S@A.T; recovered=FastICA(n_components=2,random_state=1,whiten='unit-variance',max_iter=1000).fit_transform(X)
fig,ax=plt.subplots(3,1,figsize=(10,6));
for a,data,title in zip(ax,(S,X,recovered),('Sources','Mixtures','ICA recovered')): a.plot(t,data); a.set_title(title)
plt.tight_layout(); plt.show()
