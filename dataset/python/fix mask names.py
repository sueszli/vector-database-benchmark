from os import rename,walk

f = []
for (dirpath, dirnames, filenames) in walk("MASKS_RESIZED"):
    f.extend(filenames)
    break

for file in f:
    rename("MASKS_RESIZED/"+file, "MASKS_RESIZED/"+file[:-4])