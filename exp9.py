"""Classify handwritten digits with a multilayer perceptron."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score
X,y=load_digits(return_X_y=True); X=X/16.
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',max_iter=300,early_stopping=True,random_state=42)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print('Accuracy:',accuracy_score(yte,pred)); print(classification_report(yte,pred))
