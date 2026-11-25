"""Compare PCA reconstruction error and explained variance on noisy data."""
import numpy as np
from sklearn.decomposition import PCA
rng=np.random.default_rng(4); latent=rng.normal(size=(500,2)); W=rng.normal(size=(2,8)); X=latent@W+rng.normal(0,.35,(500,8))
for k in (1,2,3,5):
 p=PCA(n_components=k).fit(X); Z=p.transform(X); recon=p.inverse_transform(Z)
 print(f'components={k}: reconstruction MSE={np.mean((X-recon)**2):.4f}, explained variance={p.explained_variance_ratio_.sum():.3f}')
