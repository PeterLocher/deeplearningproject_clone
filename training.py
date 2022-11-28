import os

import keras_unet.models as km
import keras_unet.utils as ku
from keras import metrics as met
from keras.optimizers import Adamax

import constants
from image_load import ImageMaskGenerator
from u_net import simple_u_net, classic_u_net

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


metrics = [met.Accuracy(name="accuracy"),
           met.metrics.MeanIoU(name="iou", num_classes=7)]


def train_km_unet(samples_per_epoch, epochs, batch_size, validate=True, num_classes=2, img_size=1024, model_type=km.custom_unet, learning_rate=0.001, single_class=-1, prefix=""):
    model = model_type(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='mse', metrics=metrics)
    history = train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="kar_mse_" + prefix,
                      validate=validate, single_class=single_class)
    ku.plot_segm_history(history, metrics=["iou", "val_iou"], losses=["loss", "val_loss"])
    return model


def train_my_unet(samples_per_epoch, epochs, batch_size, validate=True, num_classes=2, img_size=1024, learning_rate=0.001, single_class=-1, prefix=""):
    model = simple_u_net(num_classes, img_size)
    model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='mse', metrics=metrics)
    history = train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="my_unet_mse_" + prefix, validate=validate,
                      single_class=single_class)
    ku.plot_segm_history(history, metrics=["iou", "val_iou"], losses=["loss", "val_loss"])


def train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="model", validate=True, single_class=-1):
    gen = ImageMaskGenerator(shuffle=True, single_class=single_class)
    gen.set_up_as_sequence(samples_per_epoch, batch_size)
    gen.skip = True
    gen_val = None
    if validate:
        gen_val = ImageMaskGenerator(data_path=constants.validation_data_path, shuffle=True, single_class=single_class)
        gen_val.set_up_as_sequence(samples_per_epoch, batch_size)
    history = model.fit(gen, epochs=epochs, shuffle=True, verbose=1, validation_data=gen_val)
    model.save(file_prefix + "_skip_china_" + str(samples_per_epoch * epochs) + "_" + str(epochs) + "_" + str(batch_size))
    return history


train_km_unet(32, 75, 8, img_size=256, single_class=[6, 7], prefix="growth")
train_km_unet(32, 100, 8, img_size=256, single_class=[6, 7], prefix="growth")
train_km_unet(32, 125, 8, img_size=256, single_class=[6, 7], prefix="growth")
train_km_unet(32, 150, 8, img_size=256, single_class=[6, 7], prefix="growth")
train_km_unet(32, 175, 8, img_size=256, single_class=[6, 7], prefix="growth")


train_km_unet(32, 250, 8, img_size=256, single_class=3, prefix="road")
train_km_unet(32, 300, 8, img_size=256, single_class=3, prefix="road")
train_km_unet(32, 350, 8, img_size=256, single_class=3, prefix="road")

train_my_unet(32, 75, 8, img_size=256, single_class=3, prefix="road")
train_my_unet(32, 100, 8, img_size=256, single_class=3, prefix="road")
train_my_unet(32, 125, 8, img_size=256, single_class=3, prefix="road")
train_my_unet(32, 150, 8, img_size=256, single_class=3, prefix="road")
train_my_unet(32, 200, 8, img_size=256, single_class=3, prefix="road")
train_my_unet(32, 300, 8, img_size=256, single_class=3, prefix="road")

#China urban road detection
#train_km_unet(32, 64, 8, img_size=1024, single_class=3, learning_rate=0.0004)
