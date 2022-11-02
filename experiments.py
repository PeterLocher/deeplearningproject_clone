import keras
import tensorflow.python
import numpy as np
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt

import constants
from image_load import ImageMaskGenerator, one_hot_to_rgb, from_one_hot
from u_net import u_net_gray, u_net_color


def train_u_net(samples, epochs, batch_size):
    gen = ImageMaskGenerator()
    model = u_net_gray(8, 1024)
    image_data, mask_data = gen.next_samples(samples)
    print("Samples generated")
    model.fit(image_data, mask_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    model.save("model_unet_" + str(samples) + "_" + str(epochs) + "_" + str(batch_size))


def train_u_net_g(samples_per_epoch, epochs, batch_size, validate=True, grayscale=True, num_classes=8, img_size=1024):
    gen = ImageMaskGenerator(data_path=constants.training_data_path, grayscale=grayscale)
    gen.set_up_as_sequence(samples_per_epoch, batch_size)
    gen_val = None
    if validate:
        gen_val = ImageMaskGenerator(data_path=constants.validation_data_path, grayscale=grayscale)
        gen_val.set_up_as_sequence(samples_per_epoch, batch_size)
    model = (u_net_gray if grayscale else u_net_color)(num_classes, img_size)
    model.fit(gen, epochs=epochs, shuffle=True, verbose=1, validation_data=gen_val)
    model.save("model_unet_" + ("" if grayscale else "color_") + str(samples_per_epoch * epochs) + "_" + str(epochs) + "_" + str(batch_size))


def try_u_net(model, grayscale=True):
    gen = ImageMaskGenerator(data_path=constants.validation_data_path, grayscale=grayscale)
    image_test, mask_test = gen.next_samples(2)
    out = model.predict(image_test)

    img = Image.fromarray(np.squeeze(image_test[0]), mode="L") if grayscale else Image.fromarray(image_test[0], mode="RGB")
    img.save("out/" + "image.png")

    img = Image.fromarray(one_hot_to_rgb(out)[0], mode='RGB')
    img.save("out/" + "predicted_mask.png")
    plt.imshow(img, interpolation='nearest')
    plt.show()
    plt.plot()

    img = Image.fromarray(one_hot_to_rgb(mask_test)[0], mode='RGB')
    img.save("out/" + "true_mask.png")


def test_model(model, grayscale=True):
    print(model.metrics_names)
    gen = ImageMaskGenerator(data_path=constants.validation_data_path, grayscale=grayscale)
    loss = model.evaluate(gen, verbose=False)
    print('Test loss: ', loss)


#train_u_net(samples=128, epochs=30, batch_size=8)
#train_u_net_g(samples_per_epoch=64, epochs=14, batch_size=8, validate=True, grayscale=False)
#try_u_net(load_model("model_unet_896_14_8"))
#test_model(load_model("model_unet_896_14_8"))
#test_model(load_model("model_unet_color_896_14_8"), grayscale=False)
test_model(load_model("model_kar_unet_896_14_8"), grayscale=True)
