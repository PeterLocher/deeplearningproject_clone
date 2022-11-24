import os

import keras
import numpy as np
from PIL import Image
from keras.saving.save import load_model
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import constants
from image_load import one_hot_to_rgb, FastImageMaskGenerator, one_hot_to_rgb_single_class, ImageMaskGenerator

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


def try_u_net(model, grayscale=False, samples=5, one_hot_to_rgb_function=one_hot_to_rgb, seed=1, single_class=-1):
    gen = ImageMaskGenerator(data_path=constants.test_data_path, single_class=single_class)
    image_test, mask_test = gen.next_samples(samples)
    out = model.predict(image_test)
    mask_images = one_hot_to_rgb_function(mask_test)
    pred_images = one_hot_to_rgb_function(out)
    plot_images(org_imgs=image_test, mask_imgs=mask_images, pred_imgs=pred_images, grayscale=grayscale)

    # img = Image.fromarray(image_test[0], mode="RGB")
    # img.save("out/" + "image.png")
    #
    # img_mask = Image.fromarray(mask_images[0], mode='RGB')
    # img_mask.save("out/" + "true_mask.png")
    #
    # img_pred = Image.fromarray(pred_images[0], mode='RGB')
    # img_pred.save("out/" + "predicted_mask.png")


def test_model(model):
    print(model.metrics_names)
    gen = ImageMaskGenerator(data_path=constants.validation_data_path, shuffle=True)
    loss = model.evaluate(gen, verbose=False)
    print('Test loss: ', loss)


def test_to_one_hot():
    mask_one_hot = np.load("China_Rural_256/Val/masks_npy_3/2550_5.npy")
    mask = one_hot_to_rgb_single_class(np.array([mask_one_hot]))
    plot_images([np.asarray(Image.open("China_Rural_256/Val/images_png/2550_5.png"))], mask, grayscale=False)


def show_vanishing_point_of_road_full_image(pref="kar_mse_skip_road"):
    try_u_net(load_model(pref + "_64_2_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_128_4_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_192_6_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_256_8_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_320_10_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
    try_u_net(load_model(pref + "_384_12_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")


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


def show_feature_maps(model, layers=None, feature_maps_per_layer=16, image_number=0):
    if layers is None:
        layers = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6", "conv2d_7", "conv2d_8", "conv2d_9"]
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    fm_model = keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    gen = ImageMaskGenerator(data_path=constants.validation_data_path)
    gen.current_sample = image_number
    images, masks = gen.next_samples(number_of_samples=1)
    plt.imshow(Image.fromarray(images[0], mode="RGB"))
    feature_maps = fm_model.predict(images)
    rows = len(layers)
    cols = feature_maps_per_layer
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    row = 0
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if layer_name not in layers:
            continue
        axes[row, 0].set_ylabel(layer_name.replace("_", " "))
        axes[row, 0].get_yaxis().label.set_size(20)
        print(layer_name + " has shape " + str(feature_map.shape))
        k = int(min(cols, feature_map.shape[-1]))
        for col in range(k):
            feature_image = feature_map_to_image(feature_map[0, :, :, col])
            axes[row, col].imshow(feature_image)
            axes[row, col].get_xaxis().set_ticks([])
            axes[row, col].get_yaxis().set_ticks([])
        row += 1
    fig.tight_layout()
    plt.show()


def feature_map_to_image(feature_image):
    feature_image -= feature_image.mean()
    feature_image /= feature_image.std()
    feature_image *= 64
    feature_image += 128
    feature_image = np.clip(feature_image, 0, 255).astype('uint8')
    return feature_image


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
        image = Image.fromarray(weights[0, :, :, 0, i], mode='L')
        row = int(np.floor(i/cols))
        col = int(i % cols)
        axes[row, col].imshow(image)
        axes[row, col].set_axis_off()
    plt.show()


try_u_net(load_model("kar_mse_poland_skip_road_1600_50_8"), seed=0, one_hot_to_rgb_function=one_hot_to_rgb_single_class, single_class=3)
show_feature_maps(load_model("kar_mse_poland_skip_road_1600_50_8"), image_number=0)
#visualize_intermediate_layer(load_model("kar_mse_skip_road_3200_100_8"))
#show_vanishing_point_of_road("kar_mse_skip_road")
#show_vanishing_point_of_road_full_image()
#show_vanishing_point_of_road_Jaccard(train_pretrained_model)
#try_u_net(load_model("kar_mse_skip_road_3200_100_8"), one_hot_to_rgb_function=one_hot_to_rgb_single_class, mask_folder="masks_npy_3")
#model = train_pretrained_model(16, 100, 8)
#try_u_net(load_model("all_class_models/model_kar_unet_color_3200_100_8"))
#try_u_net(load_model("all_class_models/model_kar_unet_color_9600_300_8"))
#test_model(load_model("pretrained_unet_color_3200_200_4"), grayscale=False, num_classes=4)
