from __future__ import print_function
import os
from keras import backend as K, Input, Model
from keras.datasets import mnist
from tensorflow import keras
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# Utility function for showing images
def show_imgs(x_test, n=10):
    sz = x_test.shape[1]
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(sz,sz))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Pre-process inputs
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class indices to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Examples of test images')
show_imgs(x_test)

import numpy as np
from matplotlib import pyplot as plt

x_train_seg = np.zeros((5000, 64, 64, 1))
y_train_seg = np.zeros((5000, 64, 64, 10))

for i in range(5000):
    ## 1
    q = np.random.randint(0, 2)  # 1st or 2nd image quadrant?
    rand_ix = np.random.randint(0, x_train.shape[0])
    x_off_start = np.random.randint(0, 5)
    y_off_start = np.random.randint(0, 5)
    x_train_seg[i, 0 + x_off_start:28 + x_off_start, 32 * q + y_off_start:32 * q + 28 + y_off_start, :] = x_train[
                                                                                                          rand_ix, :, :,
                                                                                                          :]

    # Mask
    tmp = x_train[rand_ix, :, :, :]
    ix = np.where(tmp > 0.1)
    tmp = np.zeros(tmp.shape)
    tmp[ix] = 1
    label = np.argmax(y_train[rand_ix, :])
    y_train_seg[i, 0 + x_off_start:28 + x_off_start, 32 * q + y_off_start:32 * q + 28 + y_off_start,
    label] = tmp.squeeze()

    ## 2
    q = np.random.randint(0, 2)  # 3rd or 4th image quadrant?
    rand_ix = np.random.randint(0, x_train.shape[0])
    x_off_start = np.random.randint(0, 5)
    y_off_start = np.random.randint(0, 5)
    x_train_seg[i, 32 + x_off_start:32 + 28 + x_off_start, 32 * q + y_off_start:32 * q + 28 + y_off_start, :] = x_train[
                                                                                                                rand_ix,
                                                                                                                :, :, :]

    # Mask
    tmp = x_train[rand_ix, :, :, :]
    ix = np.where(tmp > 0.1)
    tmp = np.zeros(tmp.shape)
    tmp[ix] = 1
    label = np.argmax(y_train[rand_ix, :])
    y_train_seg[i, 32 + x_off_start:32 + 28 + x_off_start, 32 * q + y_off_start:32 * q + 28 + y_off_start,
    label] = tmp.squeeze()

for ex in range(5):
  plt.figure(figsize=(20,6))
  rand_ix = np.random.randint(0,5000)
  ax = plt.subplot(1,11,1)
  plt.imshow(x_train_seg[rand_ix,:,:,:].squeeze())
  plt.gray()
  plt.title('Input image')
  for i in range(10):
    ax = plt.subplot(1,11,i+2)
    plt.imshow(y_train_seg[rand_ix,:,:,i].squeeze())
    plt.gray()
    plt.title("Out class "+str(i))
print(x_train_seg.shape, y_train_seg.shape)

from keras.layers import concatenate, BatchNormalization, Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.activations import softmax

# See last layer of network
def softMaxAxis3(x):
    return softmax(x,axis=3)

def my_conv(x,filters,kernel_size=3,padding='same',kernel_initializer='he_normal'):
  x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

inputs = Input(shape=(64, 64, 1))

# Encoder
conv1 = my_conv(inputs,filters=8)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = my_conv(pool1,filters=16)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = my_conv(pool2,filters=32)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = my_conv(pool3,filters=64)

# Decoder
up7 = my_conv(conv4,filters=32)
up7 = UpSampling2D(size = (2,2))(up7)
merge7 = concatenate([conv3,up7], axis = 3)
up8 = my_conv(merge7,filters=16)
up8 = UpSampling2D(size = (2,2))(up8)
merge8 = concatenate([conv2,up8], axis = 3)
up9 = my_conv(merge8,filters=8)
up9 = UpSampling2D(size = (2,2))(up9)
merge9 = concatenate([conv1,up9], axis = 3)

# Perform softmax on each pixel, so axis should be 3 because output has shape: batch_size x 64 x 64 x num_classes
conv11 = Conv2D(num_classes, 1, activation = softMaxAxis3)(merge9)

model = Model(inputs, conv11)
model.summary()
model.compile(optimizer = keras.optimizers.RMSprop(learning_rate = 0.01), loss = 'mse')

model.fit(x_train_seg, y_train_seg, epochs=20, batch_size=64, shuffle=True, verbose=1)
print("TRAINING DONE")

# Pick 4 random examples
rand_ix = np.random.randint(0,5000,4)
out = model.predict(x_train_seg[rand_ix,:,:,:])
ref = y_train_seg[rand_ix,:,:,:].squeeze()
for k in range(4):
  plt.figure(figsize=(20,4))
  plt.subplot(2,11,1)
  plt.imshow(x_train_seg[rand_ix[k],:,:,:].squeeze())
  plt.title('Input image')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  for i in range(10):
    ax = plt.subplot(2,11,i+2)
    plt.imshow(out[k,:,:,i].squeeze(),vmin=0,vmax=1)
    plt.title('Predicted ' + str(i))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2,11,11+i+2)
    plt.imshow(ref[k,:,:,i].squeeze(),vmin=0,vmax=1)
    plt.title('Ground truth ' + str(i))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)