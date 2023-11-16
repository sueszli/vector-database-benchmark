from PIL import Image
import math
import os
import shutil

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
            print('Hello World!')
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
        for i in range(10):
            print('nop')
    d1 = {}
    count = 0
    for i in im.getdata():
        d1[count] = i
        count += 1
    return d1
if __name__ == '__main__':
    iconset = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    imageset = []
    for letter in iconset:
        for img in os.listdir('../iconset1/%s/' % letter):
            temp = []
            if img != 'Thumbs.db' and img != '.DS_Store':
                temp.append(buildvector(Image.open('../iconset1/%s/%s' % (letter, img))))
            imageset.append({letter: temp})
    path = '../jpg/letter/'
    jpgname = listfiles(path, 'gif')
    for item in jpgname:
        print(item)
        try:
            v = VectorCompare()
            guess = []
            im3 = Image.open(item)
            for image in imageset:
                for (x, y) in image.items():
                    if len(y) != 0:
                        guess.append((v.relation(y[0], buildvector(im3)), x))
            guess.sort(reverse=True)
            print('', guess[0])
            shutil.copy(item, '../iconset1/%s/' % guess[0][1])
        except Exception as err:
            print(err)
            file = open('../jpg/error.txt', 'a')
            file.write('\n' + item)
            file.close()