import os
import numpy as np
from PIL import Image, ImageOps


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
                index = np.argmax(masks[sample, x, y])*30
                label_masks[sample, x, y] = index
    return np.uint8(label_masks.astype(int))


def one_hot_to_rgb(masks):
    samples, w, h, channels = masks.shape
    label_masks = np.zeros((samples, w, h, 3))
    for sample in range(samples):
        for x in range(w):
            for y in range(h):
                index = np.argmax(masks[sample, x, y])*30
                label_masks[sample, x, y] = (index, index, index)
    return np.uint8(label_masks.astype(int))


class ImageMaskGenerator:
    image_path = "P:/SatelliteData/China/Rural/images_png"
    mask_path = "P:/SatelliteData/China/Rural/masks_png"

    image_names = os.listdir(image_path + "/img")
    current_sample = 0

    def next_samples(self, number_of_samples=8):
        images, masks = [], []
        images_to_load = self.image_names[self.current_sample:self.current_sample + number_of_samples]
        self.current_sample += number_of_samples
        for image_name in images_to_load:
            image = np.asarray(ImageOps.grayscale(Image.open(self.image_path + "/img/" + image_name)))
            mask = np.asarray(Image.open(self.mask_path + "/img/" + image_name))
            images.append(image)
            masks.append(mask)
        images = np.asarray(images)
        images = np.expand_dims(images, axis=3)
        masks = np.asarray(masks)
        masks = to_one_hot(masks)
        print(images_to_load)
        return images, masks
