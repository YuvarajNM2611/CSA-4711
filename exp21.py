"""Recover interpretable latent factors from correlated synthetic measures."""
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
rng=np.random.default_rng(9); factors=rng.normal(size=(600,2))
loadings=np.array([[.9,.1],[.8,.2],[.7,.1],[.1,.9],[.2,.8],[.1,.7]])
X=factors@loadings.T+rng.normal(0,.25,(600,6)); Z=StandardScaler().fit_transform(X)
fa=FactorAnalysis(n_components=2,random_state=9).fit(Z)
print('Rows are observed features; columns are inferred factors:
',fa.components_.T.round(3))
print('Noise variance per feature:',fa.noise_variance_.round(3))
