"""Fit PCA to Iris data and report explained variance."""
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X,y=load_iris(return_X_y=True); Z=StandardScaler().fit_transform(X); pca=PCA().fit(Z)
print('Explained variance ratio:',pca.explained_variance_ratio_.round(4))
print('Cumulative:',pca.explained_variance_ratio_.cumsum().round(4))
print('Components for >=95% variance:',(pca.explained_variance_ratio_.cumsum()<.95).sum()+1)
