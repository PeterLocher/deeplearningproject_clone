from random import Random
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

import constants

image_path = constants.data_path + "/images_png"
mask_path = constants.data_path + "/masks_png"


def preprocess_image(image):
    prepped_img = preprocess_input(image[0:64, 0:64])
    gray_prepped_img = tf.image.rgb_to_grayscale(prepped_img)
    print(gray_prepped_img.shape)
    return gray_prepped_img


def preprocess_mask(mask):
    prepped_mask = preprocess_input(mask)
    prepped_mask = prepped_mask[0:64, 0:64]
    w, h, rgb = prepped_mask.shape
    prepped_mask = np.concatenate((prepped_mask, np.zeros((w, h, 8 - rgb))), axis=2)
    for x in range(w):
        for y in range(h):
            one_hot = np.zeros(8)
            one_hot[int(prepped_mask[x, y, 0])] = 1
            prepped_mask[x, y] = one_hot
    print(prepped_mask.shape)
    return prepped_mask


seed = Random().randint(0, 9999999)
image_generator = ImageDataGenerator(preprocessing_function=preprocess_image).flow_from_directory(
    image_path, target_size=(64, 64), batch_size=2, seed=seed)
mask_generator = ImageDataGenerator(preprocessing_function=preprocess_mask).flow_from_directory(
    mask_path, target_size=(64, 64), batch_size=2, seed=seed)

china_rural_gen = zip(image_generator, mask_generator)
