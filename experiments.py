from keras import metrics as met
import tensorflow.python
import numpy as np
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt
import keras_unet.utils as ku

import constants
from image_load import ImageMaskGenerator, one_hot_to_rgb, from_one_hot
from u_net import u_net_gray, u_net_color

metrics = [met.Accuracy(), met.MeanSquaredError(name="mse")]


def train_u_net(samples, epochs, batch_size):
    gen = ImageMaskGenerator()
    model = u_net_gray(8, 1024,metrics=metrics)
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
    model = (u_net_gray if grayscale else u_net_color)(num_classes, img_size, metrics=metrics)
    history = model.fit(gen, epochs=epochs, shuffle=True, verbose=1, validation_data=gen_val)
    model.save("model_unet_" + ("" if grayscale else "color_") + str(samples_per_epoch * epochs) + "_" + str(epochs) + "_" + str(batch_size))
    ku.plot_segm_history(history, metrics=["accuracy", "mse"], losses=["mse"])


def plot_images(org_imgs, mask_imgs, pred_imgs=None, grayscale=True, figsize=4):
    n_images = len(org_imgs)
    cols = 3
    fig, axes = plt.subplots(n_images, cols, figsize=(cols * figsize, n_images * figsize), squeeze=False)
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for i in range(0, n_images):
        img = Image.fromarray(np.squeeze(org_imgs[i]), mode="L") if grayscale else Image.fromarray(org_imgs[i], mode="RGB")
        axes[i, 0].imshow(img, interpolation='nearest')
        axes[i, 0].set_axis_off()
        img_mask = Image.fromarray(mask_imgs[i], mode='RGB')
        axes[i, 1].imshow(img_mask)
        axes[i, 1].set_axis_off()
        axes[i, 2].set_axis_off()
        if pred_imgs is not None:
            img_pred = Image.fromarray(pred_imgs[i], mode='RGB')
            axes[i, 2].imshow(img_pred)
    plt.show()
    plt.plot()


def try_u_net(model, grayscale=True):
    gen = ImageMaskGenerator(data_path=constants.validation_data_path, grayscale=grayscale)
    image_test, mask_test = gen.next_samples(5)
    out = model.predict(image_test)
    mask_images = one_hot_to_rgb(mask_test)
    pred_images = one_hot_to_rgb(out)
    plot_images(org_imgs=image_test, mask_imgs=mask_images, pred_imgs=pred_images, grayscale=grayscale)

    img = Image.fromarray(np.squeeze(image_test[0]), mode="L") if grayscale else Image.fromarray(image_test[0], mode="RGB")
    img.save("out/" + "image.png")

    img_mask = Image.fromarray(mask_images[0], mode='RGB')
    img_mask.save("out/" + "true_mask.png")

    img_pred = Image.fromarray(pred_images[0], mode='RGB')
    img_pred.save("out/" + "predicted_mask.png")


def test_model(model, grayscale=True):
    print(model.metrics_names)
    gen = ImageMaskGenerator(data_path=constants.validation_data_path, grayscale=grayscale)
    loss = model.evaluate(gen, verbose=False)
    print('Test loss: ', loss)


#train_u_net(samples=128, epochs=30, batch_size=8)
#train_u_net_g(samples_per_epoch=64, epochs=14, batch_size=8, validate=True, grayscale=False)
try_u_net(load_model("model_kar_poland_unet_color_1984_31_8"), grayscale=False)
#test_model(load_model("model_unet_896_14_8"))
#test_model(load_model("model_unet_color_896_14_8"), grayscale=False)
#test_model(load_model("model_kar_poland_unet_color_896_14_8"), grayscale=False)
#test_model(load_model("model_kar_poland_unet_color_1984_31_8"), grayscale=False)
