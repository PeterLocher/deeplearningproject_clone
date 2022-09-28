import os
from random import Random

from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_path = "P:/SatelliteData/China/Rural/images_png"
mask_path = "P:/SatelliteData/China/Rural/masks_png"

prepped_image_path = "ChinaPrepped/Rural/images_png"
prepped_mask_path = "ChinaPrepped/Rural/masks_png"

samples = 8

image_names = os.listdir(image_path + "/img")[0:samples]
images = []
masks = []

for image_name in image_names:
    image = np.asarray(ImageOps.grayscale(Image.open(image_path + "/img/" + image_name)))[0:508, 0:508]
    mask = np.asarray(Image.open(mask_path + "/img/" + image_name))[0:508, 0:508]
    images.append(image)
    masks.append(mask)


for index, image in enumerate(images):
    img = Image.fromarray(image)
    img.save(prepped_image_path + "/img/" + image_names[index])

for index, mask in enumerate(masks):
    img = Image.fromarray(mask)
    img.save(prepped_mask_path + "/img/" + image_names[index])
