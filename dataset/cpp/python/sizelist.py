#!/usr/bin/python3
# -*- coding: utf-8 -*-

# データリストファイルの作成

import os.path
import sys

from PIL import Image

data_path = '/Users/yuka/Desktop/resarch/pi_system/raspi/dataDir/'
list_path = '/Users/yuka/Desktop/resarch/pi_system/raspi/img_data.txt'


if __name__ == "__main__":

    while True:

        # 画像があるディレクトリ一覧
        datalist = os.listdir(data_path)

        with open(list_path, mode='w') as f:

            # 昇順に並べ替え
            datalist = sorted(datalist)

            # データサイズの取得
            size = os.path.getsize(str(datalist))
            #size = os.path.getsize()
            print(size)

            # データリストの書き込み
            f.write('\n'.join(str(size)))

            with open(list_path) as f:
                print(f.read())
