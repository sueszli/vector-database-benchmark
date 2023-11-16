from PIL import Image
import time
import random
import os
from PIL import Image

def listfiles(rootdir, prefix='.xml'):
    if False:
        while True:
            i = 10
    file = []
    for (parent, dirnames, filenames) in os.walk(rootdir):
        if parent == rootdir:
            for filename in filenames:
                if filename.endswith(prefix):
                    file.append(rootdir + filename)
            return file
        else:
            pass

def createjia(path):
    if False:
        return 10
    try:
        os.makedirs(path)
    except:
        pass
if __name__ == '__main__':
    path = '../jpg/img/'
    jpgname = listfiles(path, 'jpg')
    for item in jpgname:
        try:
            jpgpath = item
            im = Image.open(jpgpath)
            im = im.convert('P')
            his = im.histogram()
            values = {}
            for i in range(0, 256):
                values[i] = his[i]
            temp = sorted(values.items(), key=lambda x: x[1], reverse=True)
            im2 = Image.new('P', im.size, 255)
            for y in range(im.size[1]):
                for x in range(im.size[0]):
                    pix = im.getpixel((x, y))
                    if pix == 0:
                        im2.putpixel((x, y), 0)
            inletter = False
            foundletter = False
            start = 0
            end = 0
            letters = []
            for x in range(im2.size[0]):
                for y in range(im2.size[1]):
                    pix = im2.getpixel((x, y))
                    if pix != 255:
                        inletter = True
                if foundletter == False and inletter == True:
                    foundletter = True
                    start = x
                if foundletter == True and inletter == False:
                    foundletter = False
                    end = x
                    letters.append((start, end))
                inletter = False
            count = 0
            for letter in letters:
                im3 = im2.crop((letter[0], 0, letter[1], im2.size[1]))
                a = random.randint(0, 10000)
                im3.save('../jpg/letter/%s.gif' % (time.strftime('%Y%m%d%H%M%S', time.localtime()) + str(a)))
                count += 1
        except Exception as err:
            print(err)
            file = open(',,/jpg/error.txt', 'a')
            file.write('\n' + item)
            file.close()