import os
import random

import numpy as np
from PIL import Image
import constants
from image_load import to_one_hot


def prep_images_and_masks(path=constants.training_data_path):
    image_path = path + "/images_png"
    mask_path = path + "/masks_png"

    prepped_image_path = path + "/images_npy"
    prepped_mask_path = path + "/masks_npy"

    if "images_npy" not in os.listdir(path):
        os.mkdir(prepped_image_path)
        os.mkdir(prepped_mask_path)

    image_names = os.listdir(image_path)

    for image_name in image_names:
        print(image_name)
        image = np.asarray(Image.open(image_path + "/" + image_name))
        mask_name = image_name[0:len(image_name) - 3] + "png"
        mask = np.asarray(Image.open(mask_path + "/" + mask_name))
        mask_one_hot = to_one_hot(np.array([mask]))[0]
        np.save(prepped_image_path + "/" + image_name[0:len(mask_name) - 4], image)
        np.save(prepped_mask_path + "/" + mask_name[0:len(mask_name) - 4], mask_one_hot)


#prep_images_and_masks(path=constants.training_data_path)
prep_images_and_masks(path=constants.test_data_path)
#prep_images_and_masks(path=constants.validation_data_path)