"""Compare exploding recurrent gradients before and after global-norm clipping."""
import numpy as np
rng=np.random.default_rng(7); T=30; h=0.; states=[]; xs=rng.normal(size=T)
for x in xs: h=np.tanh(1.35*h+x); states.append(h)
# BPTT scalar gradients for a loss on final state, with recurrent weight 1.35
G=[]; grad=1.
for state in reversed(states): grad*=1.35*(1-state**2); G.append(grad)
G=np.array(G); norm=np.linalg.norm(G); clipped=G*min(1.,1./(norm+1e-12))
print('Unclipped gradient norm:',norm); print('Clipped global norm:',np.linalg.norm(clipped)); print('Largest absolute gradient before/after:',np.max(np.abs(G)),np.max(np.abs(clipped)))
