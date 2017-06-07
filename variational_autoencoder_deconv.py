#coding:utf-8
# 修改为CVAE  根据标签label生成对应的图片
'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge,Merge
from keras.layers import Convolution2D, Deconvolution2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.utils import np_utils

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
nb_filters = 64
# convolution kernel size
nb_conv = 3

batch_size = 100
if K.image_dim_ordering() == 'th':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
nb_epoch = 5

x = Input(batch_shape=(batch_size,) + original_img_size)
# (100,1,28,28)
conv_1 = Convolution2D(img_chns, 2, 2, border_mode='same', activation='relu')(x)
# (100,1,28,28)
conv_2 = Convolution2D(nb_filters, 2, 2,
                       border_mode='same', activation='relu',
                       subsample=(2, 2))(conv_1)
# (100,64,14,14)  下采样 (2,2)
conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv,
                       border_mode='same', activation='relu',
                       subsample=(1, 1))(conv_2)
# (100,64,14,14)  下采样 (1,1)
conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv,
                       border_mode='same', activation='relu',
                       subsample=(1, 1))(conv_3)
# (100,64,14,14)  下采样 (1,1)
flat = Flatten()(conv_4)
# (100,12544)
hidden = Dense(intermediate_dim, activation='relu')(flat)
# (100,128)
# 均值
z_mean = Dense(latent_dim)(hidden)
# (100,2)
# log方差
z_log_var = Dense(latent_dim)(hidden)
# (100,2)

# 采样
def sampling(args):
    z_mean, z_log_var = args
    # N(0,1) 标准高斯分布
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# (100,2)

# label 加入标签信息 CVAE
label=Input(batch_shape=(batch_size,10))
merge_label=merge([z,label],mode='concat')

# we instantiate these layers separately so as to reuse them later
# 编码
decoder_hid = Dense(intermediate_dim, activation='relu')
# (100,128)
decoder_upsample = Dense(nb_filters * 14 * 14, activation='relu')
# (100,12544)

if K.image_dim_ordering() == 'th':
    output_shape = (batch_size, nb_filters, 14, 14)
else:
    output_shape = (batch_size, 14, 14, nb_filters)

# 将输入 reshape 为 (64,14,14)
decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                   output_shape,
                                   border_mode='same',
                                   subsample=(1, 1),
                                   activation='relu')
decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                   output_shape,
                                   border_mode='same',
                                   subsample=(1, 1),
                                   activation='relu')
if K.image_dim_ordering() == 'th':
    output_shape = (batch_size, nb_filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, nb_filters)
decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2,
                                          output_shape,
                                          border_mode='valid',
                                          subsample=(2, 2),
                                          activation='relu')
decoder_mean_squash = Convolution2D(img_chns, 2, 2,
                                    border_mode='valid',
                                    activation='sigmoid')
# (100,10+2)
hid_decoded = decoder_hid(merge_label)
# (100,128)
up_decoded = decoder_upsample(hid_decoded)
# (100,12544)
reshape_decoded = decoder_reshape(up_decoded)
# (100,64,14,14)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
# (100,64,14,14)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
# (100,64,14,14)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
# (100,64,29,29)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
# (100,1,28,28)

# loss=xent_loss+kl_loss
def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    # Flatten
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# input_shape: (100,1,28,28)
# output_shape: (100,1,28,28)
vae = Model([x,label], x_decoded_mean_squash)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# train the VAE on MNIST digits
# (60000,28,28)  (10000,28,28)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test,10)

# (60000,1,28,28)
print('x_train.shape:', x_train.shape)

vae.fit([x_train,Y_train], x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=([x_test,Y_test], x_test))

# build a model to project inputs on the latent space
# 输入x -> 隐层表示  z_mean
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# generator
# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

label_input=Input(shape=(10,))
merge_label=merge([decoder_input,label_input],mode='concat')

# (None,2)
_hid_decoded = decoder_hid(merge_label)
# (None,128)
_up_decoded = decoder_upsample(_hid_decoded)
# (None,12544)
_reshape_decoded = decoder_reshape(_up_decoded)
#(None,64,14,14)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
#(None,64,14,14)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
#(None,64,14,14)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
# (None,64,29,29)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
# (None,1,28,28)
generator = Model([decoder_input,label_input], _x_decoded_mean_squash)

# display a 2D manifold of the digits
n = 20  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# 目前随机采样+具体label--> 生成相关图片
# 可改进为从训练样本中学习到的label的分布中采样，再结合对应label  --> 生成相关图片
#grid_x = sorted(np.random.normal(0,1,n))
#grid_y = sorted(np.random.normal(0,1,n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # tile：将 z_sample 扩充 batch_size 倍
        # (batch_size,2)
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        # (batch_size,10)
        batch_label=np_utils.to_categorical([[(i)%10]*batch_size],10)
        x_decoded = generator.predict([z_sample,batch_label],batch_size=batch_size)\
        # (100,1,28,28)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
