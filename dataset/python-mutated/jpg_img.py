from PIL import Image
import sys
import os
import re
input_path = '/Users/yuka/Desktop/resarch/pi_system/raspi/dataDir/'
output_path = '/Users/yuka/Desktop/resarch/pi_system/raspi/jpgDir/'

def main():
    if False:
        for i in range(10):
            print('nop')
    files = os.listdir(input_path)
    Image.LOAD_TRUNCATED_IMAGES = True
    for file in files:
        if file[-4:] == '.png':
            input_im = Image.open(input_path + str(file))
            rgb_im = input_im.convert('RGB')
            (file, ext) = os.path.splitext(file)
            rgb_im.save(output_path + file + '.jpg', quality=30)
            print('transcation finished ' + str(file) + '.jpg')
if __name__ == '__main__':
    main()