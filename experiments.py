import keras
import tensorflow.python
import numpy as np
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt

from image_load import ImageMaskGenerator, one_hot_to_rgb, from_one_hot
from u_net import u_net


def train_u_net(samples, epochs, batch_size):
    gen = ImageMaskGenerator()
    model = u_net(8, 1024)
    image_data, mask_data = gen.next_samples(samples)
    print("Samples generated")
    model.fit(image_data, mask_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    model.save("model_unet_" + str(samples) + "_" + str(epochs) + "_" + str(batch_size))


def train_u_net_g(samples_per_epoch, epochs, batch_size):
    gen = ImageMaskGenerator()
    gen.set_up_as_sequence(samples_per_epoch, batch_size)
    model = u_net(8, 1024)
    model.fit(gen, epochs=epochs, shuffle=True, verbose=1)
    model.save("model_unet_g_" + str(samples_per_epoch) + "_" + str(epochs) + "_" + str(batch_size))


def try_u_net(model):
    gen = ImageMaskGenerator()
    image_test, mask_test = gen.next_samples(2)
    out = model.predict(image_test)

    img = Image.fromarray(np.squeeze(image_test[0]), mode="L")
    img.save("out/" + "image.png")

    img = Image.fromarray(one_hot_to_rgb(out)[0], mode='RGB')
    img.save("out/" + "predicted_mask.png")
    plt.imshow(img, interpolation='nearest')
    plt.show()
    plt.plot()

    img = Image.fromarray(one_hot_to_rgb(mask_test)[0], mode='RGB')
    img.save("out/" + "true_mask.png")


#train_u_net(samples=128, epochs=30, batch_size=8)
train_u_net_g(samples_per_epoch=64, epochs=14, batch_size=8)
try_u_net(load_model("model_unet_g_64_14_8"))
