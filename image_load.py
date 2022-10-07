import os
import numpy as np
from PIL import Image, ImageOps
from keras.utils import Sequence

import constants


def to_one_hot(masks):
    samples, w, h = masks.shape
    masks = np.expand_dims(masks, axis=3)
    masks = np.concatenate((masks, np.zeros((samples, w, h, 7))), axis=3)
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                one_hot = np.zeros(8)
                one_hot[int(masks[sample, x, y, 0])] = 1
                masks[sample, x, y] = one_hot
    return masks


def from_one_hot(masks):
    samples, w, h, channels = masks.shape
    label_masks = np.zeros((samples, w, h))
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                index = np.argmax(masks[sample, x, y]) * 30
                label_masks[sample, x, y] = index
    return np.uint8(label_masks.astype(int))


def one_hot_to_rgb(masks):
    samples, w, h, channels = masks.shape
    label_masks = np.zeros((samples, w, h, 3))
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                index = np.argmax(masks[sample, x, y]) * 30
                label_masks[sample, x, y] = (index, index, index)
    return np.uint8(label_masks.astype(int))


class ImageMaskGenerator(Sequence):
    img_size = 1024
    training_size = 64
    batch_size = 8

    def __init__(self, data_path=constants.training_data_path, images_folder="/images_png",
                 masks_folder="/masks_png", grayscale=True) -> None:
        super().__init__()
        self.grayscale = grayscale
        self.image_path = data_path + images_folder
        self.mask_path = data_path + masks_folder
        self.image_names = os.listdir(self.image_path)
        self.current_sample = 0

    def set_up_as_sequence(self, training_size=64, batch_size=8):
        self.batch_size = batch_size
        self.training_size = training_size

    def next_samples(self, number_of_samples=8):
        images, masks = [], []
        images_to_load = self.image_names[self.current_sample:self.current_sample + number_of_samples]
        images_found = len(images_to_load)
        if images_found < number_of_samples:
            images_to_load = images_to_load + self.image_names[0:number_of_samples - images_found]
            self.current_sample = number_of_samples - images_found
        else:
            self.current_sample += number_of_samples
        for image_name in images_to_load:
            image = Image.open(self.image_path + "/" + image_name)
            if self.grayscale:
                image = ImageOps.grayscale(image)
            image = np.asarray(image)
            mask = Image.open(self.mask_path + "/" + image_name)
            mask = np.asarray(mask)
            images.append(image)
            masks.append(mask)
        images = np.asarray(images)
        if self.grayscale:
            images = np.expand_dims(images, axis=3)
        masks = np.asarray(masks)
        masks = to_one_hot(masks)
        print(images_to_load)
        return images, masks

    def __getitem__(self, index):
        return self.next_samples(self.batch_size)

    def __len__(self):
        return self.training_size // self.batch_size
