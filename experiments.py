import os

import numpy as np
import segmentation_models as sm
from PIL import Image
from keras.saving.save import load_model
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import constants
from image_load import ImageMaskGenerator, one_hot_to_rgb, FastImageMaskGenerator


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


def try_u_net(model, grayscale=True, samples=5):
    gen = FastImageMaskGenerator(data_path=constants.test_data_path, shuffle=True)
    image_test, mask_test = gen.next_samples(samples)
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


def test_model(model):
    print(model.metrics_names)
    gen = FastImageMaskGenerator(data_path=constants.validation_data_path, shuffle=True)
    loss = model.evaluate(gen, verbose=False)
    print('Test loss: ', loss)


print(sm._KERAS_LAYERS)

#model = train_pretrained_model(16, 100, 8)
try_u_net(load_model("model_poland_unet_color_32000_1000_8"), grayscale=False, samples=7)
#test_model(load_model("pretrained_unet_color_3200_200_4"), grayscale=False, num_classes=4)
