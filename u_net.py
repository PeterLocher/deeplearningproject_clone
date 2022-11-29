import os

from keras import Input, Model
from keras.activations import softmax
from keras.layers import concatenate, BatchNormalization, Conv2D, Activation, MaxPooling2D, UpSampling2D
# See last layer of network
from keras.optimizers import RMSprop

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def softMaxAxis3(x):
    return softmax(x, axis=3)


def conv_layer(x, filters, kernel_size=3, padding='same', kernel_initializer='he_normal'):
    x = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def simple_u_net(num_classes, img_size):
    inputs = Input(shape=(img_size, img_size, 3))

    # Encoder
    conv1 = conv_layer(inputs, filters=8)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_layer(pool1, filters=16)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_layer(pool2, filters=32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_layer(pool3, filters=64)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_layer(pool4, filters=128)
    #pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    #conv6 = conv_layer(pool5, filters=256)
    #pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    #conv7 = conv_layer(pool6, filters=512)

    # Decoder
    up4 = conv_layer(conv5, filters=64)
    up4 = UpSampling2D(size=(2, 2))(up4)
    #merge4 = concatenate([conv6, up4], axis=3)
    #up5 = conv_layer(merge4, filters=128)
    #up5 = UpSampling2D(size=(2, 2))(up5)
    #merge5 = concatenate([conv5, up4], axis=3)
    #up6 = conv_layer(merge5, filters=64)
    #up6 = UpSampling2D(size=(2, 2))(up6)
    merge6 = concatenate([conv4, up4], axis=3)
    up7 = conv_layer(merge6, filters=32)
    up7 = UpSampling2D(size=(2, 2))(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    up8 = conv_layer(merge7, filters=16)
    up8 = UpSampling2D(size=(2, 2))(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    up9 = conv_layer(merge8, filters=8)
    up9 = UpSampling2D(size=(2, 2))(up9)
    merge9 = concatenate([conv1, up9], axis=3)

    # Perform softmax on each pixel, so axis should be 3 because output has shape: batch_size x 64 x 64 x num_classes
    conv11 = Conv2D(num_classes, 1, activation=softMaxAxis3)(merge9)

    model = Model(inputs, conv11)
    model.summary()
    return model


def classic_u_net(num_classes, img_size):
    inputs = Input(shape=(img_size, img_size, 3))

    # Encoder
    conv1 = conv_layer(inputs, filters=8)
    conv1_ = conv_layer(conv1, filters=8)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_)
    conv2 = conv_layer(pool1, filters=16)
    conv2_ = conv_layer(conv2, filters=16)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_)
    conv3 = conv_layer(pool2, filters=32)
    conv3_ = conv_layer(conv3, filters=32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_)
    conv4 = conv_layer(pool3, filters=64)
    conv4_ = conv_layer(conv4, filters=64)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_)
    conv5 = conv_layer(pool4, filters=128)

    # Decoder
    up4 = conv_layer(conv5, filters=64)
    up4 = UpSampling2D(size=(2, 2))(up4)
    merge6 = concatenate([conv4, up4], axis=3)
    up7 = conv_layer(merge6, filters=32)
    up7 = conv_layer(up7, filters=32)
    up7 = UpSampling2D(size=(2, 2))(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    up8 = conv_layer(merge7, filters=16)
    up8 = conv_layer(up8, filters=16)
    up8 = UpSampling2D(size=(2, 2))(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    up9 = conv_layer(merge8, filters=8)
    up9 = conv_layer(up9, filters=8)
    up9 = UpSampling2D(size=(2, 2))(up9)
    merge9 = concatenate([conv1, up9], axis=3)

    # Perform softmax on each pixel, so axis should be 3 because output has shape: batch_size x 64 x 64 x num_classes
    conv11 = Conv2D(num_classes, 1, activation=softMaxAxis3)(merge9)

    model = Model(inputs, conv11)
    model.summary()
    return model
