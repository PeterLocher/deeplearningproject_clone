import random
import shutil
from os import listdir

path = "P:/PyCharm/deeplearningproject/Poland_1024"
im_path = path + "/images_png"
ma_path = path + "/masks_png"

#for file_name in listdir(path):
#    source = path + "/" + file_name
#    target = path + "/" + file_name[0:len(file_name) - 6] + ".png"
#    rename(source, target)


def data_test_split():
    image_files = listdir(im_path)
    validation_files = random.sample(image_files, int(len(image_files) * 0.1))
    training_files = [file for file in image_files if file not in validation_files]
    test_files = random.sample(training_files, int(len(image_files) * 0.1))
    training_files = [file for file in training_files if file not in test_files]
    print(len(image_files), len(validation_files), len(training_files), len(test_files))

    for file_name in validation_files:
         shutil.copy(im_path + "/" + file_name, path + "/Val/images_png/" + file_name)
         mask_file = file_name[:len(file_name) - 3] + "png"
         shutil.copy(ma_path + "/" + mask_file, path + "/Val/masks_png/" + mask_file)
    for file_name in training_files:
        shutil.copy(im_path + "/" + file_name, path + "/Train/images_png/" + file_name)
        mask_file = file_name[:len(file_name) - 3] + "png"
        shutil.copy(ma_path + "/" + mask_file, path + "/Train/masks_png/" + mask_file)
    for file_name in test_files:
        shutil.copy(im_path + "/" + file_name, path + "/Test/images_png/" + file_name)
        mask_file = file_name[:len(file_name) - 3] + "png"
        shutil.copy(ma_path + "/" + mask_file, path + "/Test/masks_png/" + mask_file)
