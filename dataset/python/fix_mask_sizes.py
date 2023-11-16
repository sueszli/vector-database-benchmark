import numpy as np
# from matplotlib import pyplot as plt
from scipy.misc import imsave,imread
from scipy.misc import imresize
from os import rename,walk

f = []
for (dirpath, dirnames, filenames) in walk("MASKS_RESIZED"):
    f.extend(filenames)
    break

for file in f:
    a = plt.imread("data_road/testing/image_2/"+file).shape[0:2]
    b = imresize(plt.imread("MASKS_RESIZED/"+file),a)
    b = np.round(b/255).astype('uint8')
    b = b*255
    imsave("MASKS_RESIZED/"+file,b)
    