from os import rename,walk

f = []
for (dirpath, dirnames, filenames) in walk("DISPS"):
    f.extend(filenames)
    break


for file in f:
    if file[-3:] == 'png':
        a =2
        rename("DISPS/"+file, "DISPS/"+file[:-len('_left_disparity.png')]+'.png')