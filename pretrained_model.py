import segmentation_models as sm
import keras_unet.utils as ku
import constants
from image_load import ImageMaskGenerator
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# load your data
#x_train, y_train, x_val, y_val =

# preprocess input
#x_train = preprocess_input(x_train)
#x_val = preprocess_input(x_val)
classes = 4
samples_per_epoch = 8
batch_size = 8
epochs = 100
grayscale = False

# define model
model = sm.Unet(BACKBONE, classes=4, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
gen = ImageMaskGenerator(data_path=constants.training_data_path, grayscale=grayscale, classes=classes)
gen.set_up_as_sequence(samples_per_epoch, batch_size)
history = model.fit(gen, epochs=epochs, shuffle=True, verbose=1)
model.save("model_unet_" + ("" if grayscale else "color_") + str(samples_per_epoch * epochs) + "_" + str(epochs) + "_" + str(batch_size))
ku.plot_segm_history(history,  metrics=["val_accuracy"], losses=["mse", "val_loss"])
