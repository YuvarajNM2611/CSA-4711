"""Compare a VAE and GAN on Fashion-MNIST using simple reconstruction/sample stats."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
(x,_),_=keras.datasets.fashion_mnist.load_data(); x=x[:12000].astype('float32')/255.; x=x[...,None]; flat=28*28; zdim=16
inp=keras.Input((28,28,1)); h=layers.Flatten()(inp); h=layers.Dense(128,activation='relu')(h); mu=layers.Dense(zdim)(h); logv=layers.Dense(zdim)(h)
def sample(args):
 m,l=args; return m+keras.backend.random_normal(keras.backend.shape(m))*keras.backend.exp(.5*l)
z=layers.Lambda(sample)([mu,logv]); dec=keras.Sequential([layers.Input((zdim,)),layers.Dense(128,activation='relu'),layers.Dense(flat,activation='sigmoid'),layers.Reshape((28,28,1))]); out=dec(z)
vae=keras.Model(inp,out); vae.add_loss(keras.backend.mean(keras.backend.binary_crossentropy(inp,out))*flat+keras.backend.mean(-.5*(1+logv-keras.backend.square(mu)-keras.backend.exp(logv))))
vae.compile('adam'); vae.fit(x,epochs=3,batch_size=128,verbose=0); rec=vae.predict(x[:1000],verbose=0)
print('VAE mean reconstruction BCE:',float(np.mean(keras.losses.binary_crossentropy(x[:1000],rec))))
# GAN generator sample diversity proxy; train a compact GAN briefly
latent=32; G=keras.Sequential([layers.Input((latent,)),layers.Dense(128,activation='relu'),layers.Dense(flat,activation='sigmoid'),layers.Reshape((28,28,1))]); D=keras.Sequential([layers.Input((28,28,1)),layers.Flatten(),layers.Dense(64,activation='relu'),layers.Dense(1)])
opt=keras.optimizers.Adam(1e-4); ds=__import__('tensorflow').data.Dataset.from_tensor_slices(x).shuffle(12000).batch(128)
for real in ds.take(80):
 import tensorflow as tf
 n=tf.shape(real)[0]; noise=tf.random.normal((n,latent))
 with tf.GradientTape() as tape:
  fake=G(noise); dl=tf.reduce_mean(tf.nn.softplus(-D(real)))+tf.reduce_mean(tf.nn.softplus(D(fake)))
 opt.apply_gradients(zip(tape.gradient(dl,D.trainable_variables),D.trainable_variables))
 with tf.GradientTape() as tape: gl=tf.reduce_mean(tf.nn.softplus(-D(G(noise))))
 opt.apply_gradients(zip(tape.gradient(gl,G.trainable_variables),G.trainable_variables))
s=G(__import__('tensorflow').random.normal((100,latent)),training=False).numpy(); print('GAN sample pixel standard deviation:',float(s.std()))
