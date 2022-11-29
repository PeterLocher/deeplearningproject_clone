import os
import random

import numpy as np
from PIL import Image, ImageOps
from keras.utils import Sequence

import constants


def to_one_hot(masks, classes=6):
    if len(masks.shape) == 4:
        masks = masks[:, :, :, 0]
    samples, w, h = masks.shape
    masks = np.expand_dims(masks, axis=3)
    masks = np.concatenate((masks, np.zeros((samples, w, h, classes - 1))), axis=3)
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                one_hot = np.zeros(classes)
                label = int(masks[sample, x, y, 0]) - 2
                if label != -1:
                    one_hot[label] = 1
                masks[sample, x, y] = one_hot
    return masks


def to_one_hot_single_class(masks, class_id=3):
    if not isinstance(class_id, list):
        class_id = [class_id]
    if len(masks.shape) == 4:
        masks = masks[:, :, :, 0]
    samples, w, h = masks.shape
    masks = np.expand_dims(masks, axis=3)
    masks = np.concatenate((masks, np.zeros((samples, w, h, 1))), axis=3)
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                one_hot = np.zeros(2)
                c = int(masks[sample, x, y, 0])
                if c in class_id:
                    one_hot[1] = 1
                elif c != 1:
                    one_hot[0] = 1
                masks[sample, x, y] = one_hot
    return masks


def to_one_hot_combo_classes(masks, class_groupings=None):
    if class_groupings is None:
        class_groupings = [[1], [2, 3], [4, 5, 6, 7]]
    if len(masks.shape) == 4:
        masks = masks[:, :, :, 0]
    samples, w, h = masks.shape
    masks = np.expand_dims(masks, axis=3)
    masks = np.concatenate((masks, np.zeros((samples, w, h, 1))), axis=3)
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                one_hot = np.zeros(len(class_groupings))
                c = int(masks[sample, x, y, 0])
                label = 0
                for group in class_groupings:
                    if c in group:
                        break
                    label += 1
                one_hot[label] = 1
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


def one_hot_to_rgb_single_class(masks):
    samples, w, h, channels = masks.shape
    label_masks = np.zeros((samples, w, h, 3))
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                color = np.argmax(masks[sample, x, y]) * 200
                label_masks[sample, x, y] = (color, 0, 0)
    return np.uint8(label_masks.astype(int))


class ImageMaskGenerator(Sequence):
    training_size = 64
    batch_size = 8
    skip = False
    number_of_skips = 0

    def __init__(self, data_path=constants.training_data_path, images_folder="/images_png",
                 masks_folder="/masks_png", grayscale=False, classes=6, single_class=-1, shuffle=False, seed=0) -> None:
        super().__init__()
        self.grayscale = grayscale
        self.image_path = data_path + images_folder
        self.mask_path = data_path + masks_folder
        self.image_names = os.listdir(self.image_path)
        if shuffle:
            random.seed(seed)
            random.shuffle(self.image_names)
        self.current_sample = 0
        self.classes = classes
        self.single_class = single_class

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
            mask_file_type = self.mask_path[len(self.mask_path) - 3:]
            mask = Image.open(self.mask_path + "/" + image_name[0:len(image_name) - 3] + mask_file_type)
            mask = np.asarray(mask)
            images.append(image)
            masks.append(mask)
        images = np.asarray(images)
        if self.grayscale:
            images = np.expand_dims(images, axis=3)
        masks = np.asarray(masks)
        if self.single_class == -1:
            masks = to_one_hot(masks, classes=self.classes)
        else:
            masks = to_one_hot_single_class(masks, class_id=self.single_class)
        class_pixels = masks[:, :, :, 1].sum()
        if class_pixels < 4000 and self.skip:
            self.number_of_skips += 1
            print(class_pixels)
            print(f"skipped {self.number_of_skips} batch(es) without class {self.single_class}")
            return self.next_samples(number_of_samples=number_of_samples)
        return images, masks

    def __getitem__(self, index):
        return self.next_samples(self.batch_size)

    def __len__(self):
        return self.training_size // self.batch_size
