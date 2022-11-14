import os

import keras_unet.models as km
import keras_unet.utils as ku
import segmentation_models as sm
from keras import metrics as met
from keras.optimizers import RMSprop

import constants
from image_load import ImageMaskGenerator, FastImageMaskGenerator
from u_net import u_net_color, u_net_gray

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


metrics = [met.Accuracy(name="accuracy", dtype=None)]


def train_u_net(samples, epochs, batch_size):
    gen = ImageMaskGenerator()
    model = u_net_gray(8, 1024)
    image_data, mask_data = gen.next_samples(samples)
    print("Samples generated")
    model.fit(image_data, mask_data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    model.save("model_unet_" + str(samples) + "_" + str(epochs) + "_" + str(batch_size))


def train_pretrained_model(samples_per_epoch, epochs, batch_size, num_classes=7):
    BACKBONE = 'resnet34'
    model = sm.FPN(BACKBONE, classes=num_classes, encoder_weights='imagenet')
    model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
    history = train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="pretrained_unet", num_classes=num_classes)
    ku.plot_segm_history(history, metrics=["iou_score"], losses=["loss", "val_loss"])
    return model


def train_km_unet(samples_per_epoch, epochs, batch_size, validate=True, num_classes=7, img_size=1024, model_type=km.custom_unet):
    model = model_type(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mse', metrics=metrics)
    history = train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="model_kar_unet",
                      validate=validate, num_classes=num_classes, img_size=img_size)
    ku.plot_segm_history(history, metrics=["accuracy"], losses=["loss", "val_loss"])


def train_my_unet(samples_per_epoch, epochs, batch_size, validate=True, num_classes=7, img_size=1024):
    model = u_net_color(num_classes, img_size)
    model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mse', metrics=metrics)
    history = train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="model_unet", validate=validate, num_classes=num_classes, img_size=img_size)
    ku.plot_segm_history(history, metrics=["accuracy"], losses=["loss", "val_loss"])


def train_g(model, samples_per_epoch, epochs, batch_size, file_prefix="model", validate=True, num_classes=7, img_size=1024):
    gen = FastImageMaskGenerator(data_path=constants.training_data_path, shuffle=True)
    gen.set_up_as_sequence(samples_per_epoch, batch_size)
    gen_val = None
    if validate:
        gen_val = FastImageMaskGenerator(data_path=constants.validation_data_path, shuffle=True)
        gen_val.set_up_as_sequence(samples_per_epoch, batch_size)
    history = model.fit(gen, epochs=epochs, shuffle=True, verbose=1, validation_data=gen_val)
    model.save(file_prefix + "_color_" + str(samples_per_epoch * epochs) + "_" + str(epochs) + "_" + str(batch_size))
    return history


#train_u_net(samples=128, epochs=30, batch_size=8)
train_my_unet(32, 1000, 8, img_size=256)
train_km_unet(32, 1000, 8, img_size=256)
#train_km_unet(4, 2, 2)
#train_pretrained_model(16, 110, 4)
#train_pretrained_model(16, 200, 4)
