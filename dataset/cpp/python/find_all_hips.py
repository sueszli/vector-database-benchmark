from hipdatatool import HipMarker
import os
import shutil

path = "../data/finish-line/bmps/"

skip = 0

for filename in sorted(os.listdir(path)):
    if (filename.endswith(".bmp")) and skip == 0:
        name = filename[:-4]
        h = HipMarker(path, name, ".bmp")
        r = h.runImage()
        if r == 1:
            break
        if r == 2:
            skip = 10
    elif (filename.endswith(".bmp")) and skip > 0:
        shutil.move(path+filename, path + "empty/" + filename)
        skip -= 1
