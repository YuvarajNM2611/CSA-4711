"""Compare SimpleRNN and LSTM on the same synthetic forecasting data."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
rng=np.random.default_rng(5); s=np.sin(np.arange(750)/14)+rng.normal(0,.05,750); w=25
X=np.array([s[i:i+w] for i in range(len(s)-w)])[...,None]; y=s[w:]; cut=550
for kind in ('SimpleRNN','LSTM'):
 cell=getattr(layers,kind)(24); m=keras.Sequential([layers.Input((w,1)),cell,layers.Dense(1)])
 m.compile('adam','mse'); m.fit(X[:cut],y[:cut],epochs=8,batch_size=32,verbose=0)
 print(kind,'test MSE:',m.evaluate(X[cut:],y[cut:],verbose=0))
