"""Compare model complexity and train/test error on a synthetic nonlinear dataset."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
rng=np.random.default_rng(7)
X=np.linspace(-3,3,120).reshape(-1,1); y=np.sin(X[:,0])+rng.normal(0,.25,len(X))
train=np.arange(0,120,2); test=np.arange(1,120,2)
for degree in (1,3,10,25):
    model=make_pipeline(PolynomialFeatures(degree),LinearRegression()).fit(X[train],y[train])
    print(f"degree={degree:2d} train MSE={mean_squared_error(y[train],model.predict(X[train])):.3f} test MSE={mean_squared_error(y[test],model.predict(X[test])):.3f}")
    plt.plot(X[test],model.predict(X[test]),label=f"degree {degree}")
plt.scatter(X[test],y[test],s=12,c='black',label='test data'); plt.legend(); plt.title('Underfitting and overfitting'); plt.show()
