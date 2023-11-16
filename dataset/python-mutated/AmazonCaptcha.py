from PIL import Image
import math
import os
from concurrent.futures import ThreadPoolExecutor

def listfiles(rootdir, prefix='.xml'):
    if False:
        print('Hello World!')
    file = []
    for (parent, dirnames, filenames) in os.walk(rootdir):
        if parent == rootdir:
            for filename in filenames:
                if filename.endswith(prefix):
                    file.append(rootdir + filename)
            return file
        else:
            pass

class VectorCompare:

    def magnitude(self, concordance):
        if False:
            return 10
        total = 0
        for (word, count) in concordance.items():
            total += count ** 2
        return math.sqrt(total)

    def relation(self, concordance1, concordance2):
        if False:
            while True:
                i = 10
        topvalue = 0
        for (word, count) in concordance1.items():
            if word in concordance2:
                topvalue += count * concordance2[word]
        return topvalue / (self.magnitude(concordance1) * self.magnitude(concordance2))

def buildvector(im):
    if False:
        return 10
    d1 = {}
    count = 0
    for i in im.getdata():
        d1[count] = i
        count += 1
    return d1
path = '../jpg/img/'
iconset = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
imageset = []
for letter in iconset:
    for img in os.listdir('../iconset1/%s/' % letter):
        temp = []
        if img != '':
            temp.append(buildvector(Image.open('../iconset1/%s/%s' % (letter, img))))
        imageset.append({letter: temp})

def main(item):
    if False:
        for i in range(10):
            print('nop')
    try:
        newjpgname = []
        im = Image.open(item)
        print(item)
        im = im.convert('P')
        his = im.histogram()
        values = {}
        for i in range(0, 256):
            values[i] = his[i]
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
        v = VectorCompare()
        count = 0
        for letter in letters:
            im3 = im2.crop((letter[0], 0, letter[1], im2.size[1]))
            guess = []
            for image in imageset:
                for (x, y) in image.items():
                    if len(y) != 0:
                        guess.append((v.relation(y[0], buildvector(im3)), x))
            guess.sort(reverse=True)
            print('', guess[0])
            newjpgname.append(guess[0][1])
            count += 1
        newname = str(''.join(newjpgname))
        os.rename(item, path + newname + '.jpg')
    except Exception as err:
        print(err)
        file = open('../jpg/error.txt', 'a')
        file.write('\n' + item)
        file.close()

def runthreading():
    if False:
        for i in range(10):
            print('nop')
    pool = ThreadPoolExecutor(5)
    jpgname = listfiles(path, 'jpg')
    for item in jpgname:
        if len(item) > 30:
            pool.submit(main, item)
if __name__ == '__main__':
    runthreading()