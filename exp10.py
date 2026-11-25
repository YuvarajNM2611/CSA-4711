"""Compare regularization strategies on sklearn's digits dataset."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X,y=load_digits(return_X_y=True); X=X/16.; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=3,stratify=y)
settings=[('baseline',0,False),('L2',.01,False),('early stopping',.0001,True),('L2 + early stop',.01,True)]
for name,alpha,early in settings:
 m=MLPClassifier(hidden_layer_sizes=(100,),alpha=alpha,early_stopping=early,max_iter=300,random_state=3)
 m.fit(Xtr,ytr); print(f'{name:16} accuracy={accuracy_score(yte,m.predict(Xte)):.3f}; epochs={m.n_iter_}')
