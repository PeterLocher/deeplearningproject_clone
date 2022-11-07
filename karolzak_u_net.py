import os

import keras_unet.models as km
import matplotlib.pyplot as plt
from keras import metrics as met
from keras.optimizers import RMSprop
import constants
from image_load import ImageMaskGenerator
import keras_unet.utils as ku

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_u_net_g(samples_per_epoch, epochs, batch_size, validate=True, grayscale=True, num_classes=8, img_size=1024, model_type=km.custom_unet):
    gen = ImageMaskGenerator(data_path=constants.training_data_path, grayscale=grayscale)
    gen.set_up_as_sequence(samples_per_epoch, batch_size)
    gen_val = None
    if validate:
        gen_val = ImageMaskGenerator(data_path=constants.validation_data_path, grayscale=grayscale)
        gen_val.set_up_as_sequence(samples_per_epoch, batch_size)
    model = model_type(input_shape=(img_size, img_size, 1 if grayscale else 3), num_classes=num_classes)
    metrics = [met.Accuracy(), met.MeanSquaredError(name="mse")]
    model.compile(optimizer=RMSprop(learning_rate=0.01), loss='mse', metrics=metrics)
    history = model.fit(gen, epochs=epochs, shuffle=True, verbose=1, validation_data=gen_val)
    model.save("model_kar_poland_unet_" + ("" if grayscale else "color_") + str(samples_per_epoch * epochs) + "_" + str(epochs) + "_" + str(batch_size))
    ku.plot_segm_history(history, metrics=["accuracy"], losses=["mse"])


train_u_net_g(4, 4, 4, grayscale=False)
