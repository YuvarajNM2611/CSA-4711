"""Train a compact VAE and GAN on Fashion-MNIST and save generated samples."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
(x,_),_=keras.datasets.fashion_mnist.load_data()
x=(x[:20000].astype('float32')/255.)[...,None]
# Variational autoencoder: reconstruction loss plus KL divergence.
zdim=16
inp=keras.Input((28,28,1)); enc=layers.Flatten()(inp); enc=layers.Dense(128,activation='relu')(enc)
mu=layers.Dense(zdim)(enc); logvar=layers.Dense(zdim)(enc)
z=layers.Lambda(lambda q:q[0]+tf.exp(.5*q[1])*tf.random.normal(tf.shape(q[0])))([mu,logvar])
dec=keras.Sequential([layers.Input((zdim,)),layers.Dense(128,activation='relu'),layers.Dense(784,activation='sigmoid')])
recon=dec(z); outputs=layers.Concatenate()([recon,mu,logvar]); vae=keras.Model(inp,outputs)
def vae_loss(target,pred):
    rec=pred[:,:784]; m=pred[:,784:784+zdim]; lv=pred[:,784+zdim:]
    bce=keras.backend.binary_crossentropy(tf.reshape(target,(-1,784)),rec)
    kl=-.5*tf.reduce_sum(1+lv-tf.square(m)-tf.exp(lv),axis=1)
    return 784*bce+kl
vae.compile('adam',loss=vae_loss); vae.fit(x,x.reshape((-1,784)),epochs=3,batch_size=128,verbose=1)
vae_samples=dec(tf.random.normal((16,zdim))).numpy().reshape((-1,28,28,1))
# GAN with a convolutional generator and discriminator.
latent=64; G=keras.Sequential([layers.Input((latent,)),layers.Dense(7*7*128),layers.Reshape((7,7,128)),layers.Conv2DTranspose(64,4,2,padding='same',activation='relu'),layers.Conv2DTranspose(1,4,2,padding='same',activation='sigmoid')])
D=keras.Sequential([layers.Input((28,28,1)),layers.Conv2D(64,4,2,padding='same'),layers.LeakyReLU(.2),layers.Flatten(),layers.Dense(1)])
opt=keras.optimizers.Adam(1e-4); ds=tf.data.Dataset.from_tensor_slices(x).shuffle(20000).batch(128)
for epoch in range(3):
 for real in ds:
  n=tf.shape(real)[0]; noise=tf.random.normal((n,latent))
  with tf.GradientTape() as tape:
   fake=G(noise,training=True); dloss=tf.reduce_mean(tf.nn.relu(1-D(real,training=True)))+tf.reduce_mean(tf.nn.relu(1+D(fake,training=True)))
  opt.apply_gradients(zip(tape.gradient(dloss,D.trainable_variables),D.trainable_variables))
  with tf.GradientTape() as tape:
   fake=G(noise,training=True); gloss=-tf.reduce_mean(D(fake,training=True))
  opt.apply_gradients(zip(tape.gradient(gloss,G.trainable_variables),G.trainable_variables))
 print('GAN epoch',epoch+1,'D loss',float(dloss),'G loss',float(gloss))
gan_samples=G(tf.random.normal((16,latent)),training=False).numpy()
# Save 4x4 grids for visual comparison.
for name,samples in (('vae_samples.png',vae_samples),('gan_samples.png',gan_samples)):
    grid=tf.concat([tf.concat(list(samples[i:i+4]),axis=1) for i in range(0,16,4)],axis=0)
    keras.utils.save_img(name,grid)
print('Saved vae_samples.png and gan_samples.png')
