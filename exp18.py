"""Compare recurrent cell types and hidden widths on a common time series."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
s=np.sin(np.arange(600)/12); w=20; X=np.array([s[i:i+w] for i in range(len(s)-w)])[...,None]; y=s[w:]
for kind in ('SimpleRNN','LSTM','GRU'):
 for units in (16,32):
  m=keras.Sequential([layers.Input((w,1)),getattr(layers,kind)(units),layers.Dense(1)])
  m.compile('adam','mse'); m.fit(X[:450],y[:450],epochs=5,batch_size=32,verbose=0)
  print(kind,'units=',units,'test MSE=',round(m.evaluate(X[450:],y[450:],verbose=0),4))
