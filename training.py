import os

#run pip install keras-unet
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


def train_km_unet(samples_per_epoch, epochs, batch_size, validate=True, num_classes=2, img_size=1024, learning_rate=0.001, single_class=-1, prefix="", skip=True):
    model = km.custom_unet(input_shape=(img_size, img_size, 3), num_classes=num_classes, dropout=0)
    model.summary()
    #model = train_unet(model, samples_per_epoch, epochs, batch_size, validate, img_size, learning_rate, single_class, prefix="kar_mse_" + prefix, skip=skip)
    #return model


def train_my_unet(samples_per_epoch, epochs, batch_size, validate=True, num_classes=2, img_size=1024, learning_rate=0.001, single_class=-1, prefix="", model_function=simple_u_net, skip=True):
    model = model_function(num_classes, img_size)
    model = train_unet(model, samples_per_epoch, epochs, batch_size, validate, img_size, learning_rate, single_class, prefix="my_unet_mse_" + prefix, skip=skip)
    return model


def train_unet(model, samples_per_epoch, epochs, batch_size, validate=True, img_size=1024, learning_rate=0.001, single_class=-1, prefix="", skip=True):
    model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='mse', metrics=metrics)
    history = train_g(model, samples_per_epoch, epochs, batch_size, file_prefix=prefix, validate=validate,
                      single_class=single_class, skip=skip)
    ku.plot_segm_history(history, metrics=["iou", "val_iou"], losses=["loss", "val_loss"])
    return model


def train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="model", validate=True, single_class=-1, skip=True):
    gen = ImageMaskGenerator(shuffle=True, single_class=single_class)
    gen.set_up_as_sequence(samples_per_epoch, batch_size)
    gen.skip = skip
    gen_val = None
    if validate:
        gen_val = ImageMaskGenerator(data_path=constants.validation_data_path, shuffle=True, single_class=single_class)
        gen_val.set_up_as_sequence(samples_per_epoch, batch_size)
    history = model.fit(gen, epochs=epochs, shuffle=True, verbose=1, validation_data=gen_val)
    model.save(file_prefix + "_poland_" + str(epochs) + "_" + str(batch_size) + "_" + str(samples_per_epoch * epochs))
    return history


train_km_unet(32, 32, 8, img_size=1024, num_classes=6, prefix="multi_class", skip=False)
#train_km_unet(32, 48, 8, img_size=1024, num_classes=6, prefix="multi_class", skip=False)
#train_km_unet(32, 64, 8, img_size=1024, single_class=2)
