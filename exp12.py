"""Compare CNN filter counts on a small Fashion-MNIST subset."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
(x,y),_=keras.datasets.fashion_mnist.load_data(); x=x[:12000].astype('float32')[...,None]/255.; y=y[:12000]
(xtr,ytr),(xv,yv)=(x[:10000],y[:10000]),(x[10000:],y[10000:])
for filters in (16,32,64):
 m=keras.Sequential([layers.Input((28,28,1)),layers.Conv2D(filters,3,activation='relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(10,activation='softmax')])
 m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']); h=m.fit(xtr,ytr,validation_data=(xv,yv),epochs=3,batch_size=128,verbose=0)
 print(f'filters={filters:2}: val_accuracy={max(h.history["val_accuracy"]):.3f}')
