import keras
import numpy as np
from PIL import Image
from keras.saving.save import load_model
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


def test_u_net(model):
    gen = ImageMaskGenerator()
    image_test, mask_test = gen.next_samples(2)
    out = model.predict(image_test)

    img = Image.fromarray(np.squeeze(image_test[1]), mode="L")
    img.save("out/" + "image.png")

    img = Image.fromarray(one_hot_to_rgb(out)[1], mode='RGB')
    img.save("out/" + "predicted_mask.png")
    plt.imshow(img, interpolation='nearest')
    plt.show()
    plt.plot()

    img = Image.fromarray(one_hot_to_rgb(mask_test)[1], mode='RGB')
    img.save("out/" + "true_mask.png")


train_u_net(samples=140, epochs=64, batch_size=16)
#test_u_net(load_model("model"))
