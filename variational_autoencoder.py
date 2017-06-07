#coding:utf-8
# 变分自编码器 VAE
# 已添加标签信息，修改为(conditional)CVAE 条件变分自编码器
# 可根据标签信息 生成对应样本
'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda,merge,Merge
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.utils import np_utils

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50
epsilon_std = 1.0
# 
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
# (None,100,256)
# 高斯分布的均值
# (None,100,2)
z_mean = Dense(latent_dim)(h)
# 高斯分布的方差对数
z_log_var = Dense(latent_dim)(h)
# (None,100,2)

# 采样  x~N(,)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# (None,2)

label = Input(batch_shape=(batch_size, 10))
merge_label=merge([z,label],mode='concat')

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu') 
# 输入 0~1  输出也映射到 0~1  sigmoid
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(merge_label)
# (None,256)
x_decoded_mean = decoder_mean(h_decoded)
# (None,784)

# loss=xent_loss+kl_loss
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model([x,label], x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 0~1 norm
# (60000,28,28)
x_train = x_train.astype('float32') / 255.
# (10000,28,28)
x_test = x_test.astype('float32') / 255.
# (60000,28*28)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# (10000,28*28)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test,10)

# input_shape:(100,784)
# output_shape:(100,784)
vae.fit([x_train,Y_train], x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=([x_test,Y_test], x_test))

# build a model to project inputs on the latent space
# input_shape:(100,784)   output_shape:(100,2)
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
# (10000,2)  将输入784维压缩至2维
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

plt.figure(figsize=(6, 6))
# 每个样本都是两维  用点状图展示  颜色为对应的类别
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
# 生成
decoder_input = Input(shape=(latent_dim,))

label_input=Input(shape=(10,))
merge_label=merge([decoder_input,label_input],mode='concat')

# (None,2)
_h_decoded = decoder_h(merge_label)
# (None,256)
_x_decoded_mean = decoder_mean(_h_decoded)
# (None,784)
generator = Model([decoder_input,label_input], _x_decoded_mean)

# display a 2D manifold of the digits
n = 10  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
#grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
#grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

grid_x = sorted(np.random.normal(0,1,n))
grid_y = sorted(np.random.normal(0,1,n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # 根据z_sample 生成图片
        # 10*10 张图片 每行相同  1~10  
        # 可改变标签信息 输出相应的数字
        x_decoded = generator.predict([z_sample,np_utils.to_categorical([[(i+j)%10]],10)])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
