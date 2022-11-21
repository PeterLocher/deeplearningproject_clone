import os

import numpy as np
import segmentation_models as sm
from PIL import Image
from keras.saving.save import load_model
from matplotlib import pyplot as plt

from training import train_km_unet, train_pretrained_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import constants
from image_load import ImageMaskGenerator, one_hot_to_rgb, FastImageMaskGenerator, one_hot_to_rgb_single_class

plt.plot()


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


def try_u_net(model, grayscale=False, samples=5, one_hot_to_rgb_function=one_hot_to_rgb, mask_folder="masks_npy"):
    gen = FastImageMaskGenerator(data_path=constants.test_data_path, masks_folder=mask_folder, shuffle=True,
                                 seed=1)
    image_test, mask_test = gen.next_samples(samples)
    out = model.predict(image_test)
    mask_images = one_hot_to_rgb_function(mask_test)
    pred_images = one_hot_to_rgb_function(out)
    plot_images(org_imgs=image_test, mask_imgs=mask_images, pred_imgs=pred_images, grayscale=grayscale)

    # img = Image.fromarray(np.squeeze(image_test[0]), mode="L") if grayscale else Image.fromarray(image_test[0], mode="RGB")
    # img.save("out/" + "image.png")
    #
    # img_mask = Image.fromarray(mask_images[0], mode='RGB')
    # img_mask.save("out/" + "true_mask.png")
    #
    # img_pred = Image.fromarray(pred_images[0], mode='RGB')
    # img_pred.save("out/" + "predicted_mask.png")


def test_model(model):
    print(model.metrics_names)
    gen = FastImageMaskGenerator(data_path=constants.validation_data_path, shuffle=True)
    loss = model.evaluate(gen, verbose=False)
    print('Test loss: ', loss)


def test_to_one_hot():
    mask_one_hot = np.load("China_Rural_256/Val/masks_npy_3/2550_5.npy")
    mask = one_hot_to_rgb_single_class(np.array([mask_one_hot]))
    plot_images([np.asarray(Image.open("China_Rural_256/Val/images_png/2550_5.png"))], mask, grayscale=False)


def show_vanishing_point_of_road(pref="kar_mse_road"):
    try_u_net(load_model(pref + "_1600_50_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_2400_75_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_3200_100_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_4000_125_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_4800_150_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    #try_u_net(load_model(pref + "_5600_175_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")


def show_vanishing_point_of_road_Jaccard(training_function):
    model = training_function(32, 50, 8, img_size=256)
    try_u_net(model, one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    model = training_function(32, 75, 8, img_size=256)
    try_u_net(model, one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    model = training_function(32, 100, 8, img_size=256)
    try_u_net(model, one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    model = training_function(32, 125, 8, img_size=256)
    try_u_net(model, one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    model = training_function(32, 150, 8, img_size=256)
    try_u_net(model, one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")


def visualize_intermediate_layer(model):
    model.summary()
    layer = model.layers[25]
    weights = np.array(layer.get_weights())
    print(layer)
    print(weights.shape)
    rows = 10
    cols = 10
    fig, axes = plt.subplots(rows, cols)
    for i in range(0, rows * cols):
        image = Image.fromarray(weights[0, :, :, i, 0:3], mode='RGB')
        row = int(np.floor(i/cols))
        col = int(i % cols)
        axes[row, col].imshow(image)
        axes[row, col].set_axis_off()
    plt.show()


#visualize_intermediate_layer(load_model("kar_mse_skip_road_2400_75_8"))
#show_vanishing_point_of_road("kar_mse_skip_road")
#show_vanishing_point_of_road_Jaccard(train_pretrained_model)
#try_u_net(load_model("kar_mse_skip_road_3200_100_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
#model = train_pretrained_model(16, 100, 8)
#try_u_net(load_model("all_class_models/model_kar_unet_color_3200_100_8"))
#try_u_net(load_model("all_class_models/model_kar_unet_color_9600_300_8"))
#test_model(load_model("pretrained_unet_color_3200_200_4"), grayscale=False, num_classes=4)
