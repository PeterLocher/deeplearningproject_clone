import os
import random
import shutil
from os import listdir

path = "P:/PyCharm/deeplearningproject/Poland_256"
im_path = path + "/images_png"
ma_path = path + "/masks_png"


def rename_masks_poland():
    for file_name in listdir(ma_path):
        source = ma_path + "/" + file_name
        target = ma_path + "/" + file_name[0:len(file_name) - 6] + ".png"
        os.rename(source, target)


def data_test_split():
    image_files = listdir(im_path)
    validation_files = random.sample(image_files, int(len(image_files) * 0.1))
    training_files = [file for file in image_files if file not in validation_files]
    test_files = random.sample(training_files, int(len(image_files) * 0.1))
    training_files = [file for file in training_files if file not in test_files]
    print(len(image_files), len(validation_files), len(training_files), len(test_files))
    val_path = path + "/Val"
    train_path = path + "/Train"
    test_path = path + "/Test"
    os.mkdir(val_path)
    os.mkdir(train_path)
    os.mkdir(test_path)
    os.mkdir(val_path + "/images_png")
    os.mkdir(val_path + "/masks_png")
    os.mkdir(train_path + "/images_png")
    os.mkdir(train_path + "/masks_png")
    os.mkdir(test_path + "/images_png")
    os.mkdir(test_path + "/masks_png")

    for file_name in validation_files:
         shutil.copy(im_path + "/" + file_name, val_path + "/images_png/" + file_name)
         mask_file = file_name[:len(file_name) - 3] + "png"
         shutil.copy(ma_path + "/" + mask_file, val_path + "/masks_png/" + mask_file)
    for file_name in training_files:
        shutil.copy(im_path + "/" + file_name, train_path + "/images_png/" + file_name)
        mask_file = file_name[:len(file_name) - 3] + "png"
        shutil.copy(ma_path + "/" + mask_file, train_path + "/masks_png/" + mask_file)
    for file_name in test_files:
        shutil.copy(im_path + "/" + file_name, test_path + "/images_png/" + file_name)
        mask_file = file_name[:len(file_name) - 3] + "png"
        shutil.copy(ma_path + "/" + mask_file, test_path + "/masks_png/" + mask_file)


#rename_masks_poland()
data_test_split()
