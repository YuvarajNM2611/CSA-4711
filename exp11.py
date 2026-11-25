"""Train a CNN on Fashion-MNIST (downloads data on first run)."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
(xtr,ytr),(xte,yte)=keras.datasets.fashion_mnist.load_data()
xtr=xtr.astype('float32')[...,None]/255.; xte=xte.astype('float32')[...,None]/255.
model=keras.Sequential([layers.Input((28,28,1)),layers.Conv2D(32,3,activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64,3,activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128,activation='relu'),layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']); model.fit(xtr,ytr,epochs=5,batch_size=128,validation_split=.1)
print('Test metrics:',model.evaluate(xte,yte,verbose=0))
