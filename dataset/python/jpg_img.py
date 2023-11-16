#!/usr/bin/python3
# -*- coding: utf-8 -*-

# png から jpg にファイル変換
# https://qiita.com/hirohuntexp/items/05b7a81323dff7bdca9f

from PIL import Image
import sys
import os
import re

input_path = '/Users/yuka/Desktop/resarch/pi_system/raspi/dataDir/'
output_path = '/Users/yuka/Desktop/resarch/pi_system/raspi/jpgDir/'

def main():

    # /dataDir　ファイルリスト
    files = os.listdir(input_path)

    Image.LOAD_TRUNCATED_IMAGES = True

    for file in files:

        # png ファイルの呼び出し
        if file[-4:] == ".png":

            input_im = Image.open(input_path + str(file))

            rgb_im = input_im.convert('RGB')

            # png の拡張子を削除する
            file, ext = os.path.splitext(file)

            # jpg ファイルとして /jpgDIr に保存
            rgb_im.save(output_path + file + ".jpg",quality=30)

            print("transcation finished " + str(file) + ".jpg")

if __name__ == "__main__":
    main()
