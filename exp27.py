"""Learn stacked RBM features as a simple DBN-style representation."""
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X,y=load_digits(return_X_y=True); X=(X/16.>.35).astype(float); Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=4,stratify=y)
model=Pipeline([('rbm1',BernoulliRBM(n_components=128,n_iter=15,learning_rate=.03,random_state=1)),('rbm2',BernoulliRBM(n_components=64,n_iter=15,learning_rate=.03,random_state=2)),('classifier',LogisticRegression(max_iter=1000))])
model.fit(Xt,yt); print('DBN-style representation accuracy:',accuracy_score(yv,model.predict(Xv)))
