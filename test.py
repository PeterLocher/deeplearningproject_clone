import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

from image_load import to_one_hot, from_one_hot, one_hot_to_rgb

image_path = "P:/SatelliteData/China/Rural/images_png"
mask_path = "P:/SatelliteData/China/Rural/masks_png"

image_name = "2522.png"

image = np.asarray(ImageOps.grayscale(Image.open(image_path + "/img/" + image_name)))
mask = np.asarray(Image.open(mask_path + "/img/" + image_name))

masks = np.asarray([mask])

print(mask)
mask_2 = one_hot_to_rgb(to_one_hot(masks))[0]
#print(np.equal(mask_2, mask))
print(mask_2)
img = Image.fromarray(np.uint8(mask_2), mode='RGB')
img.save("out/" + "test.png")

plt.imshow(mask_2, interpolation='nearest')
plt.show()
plt.plot()

