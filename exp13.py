"""Prepare sliding windows and forecast a synthetic sequential signal."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
rng=np.random.default_rng(1); t=np.arange(1000); series=(np.sin(t/20)+.25*np.sin(t/5)+rng.normal(0,.04,len(t))).reshape(-1,1)
scaler=MinMaxScaler(); z=scaler.fit_transform(series)
window=30; X=np.array([z[i:i+window] for i in range(len(z)-window)]); y=z[window:]
split=int(.8*len(X)); model=keras.Sequential([layers.Input((window,1)),layers.SimpleRNN(32),layers.Dense(1)])
model.compile('adam','mse'); model.fit(X[:split],y[:split],epochs=8,batch_size=32,verbose=0)
pred=model.predict(X[split:],verbose=0); print('Test MSE:',np.mean((pred-y[split:])**2),'sample prediction:',scaler.inverse_transform(pred[:1])[0,0])
