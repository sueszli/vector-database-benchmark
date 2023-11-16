#!/usr/bin/python3
# -*- coding: utf-8 -*-

# その日以前のディレクトリをファイルにzip圧縮させて消すプログラム

import cv2
import glob
import os
import shutil
import time

from datetime import datetime
from time import sleep

data_path = '/home/pi/Desktop/pi_system/raspi/dataDir/'
zip_path = '/home/pi/Desktop/pi_system/raspi/zipfile/'

if __name__ == '__main__':

    while True:

        # 1分おきに実行
        time.sleep(60)

        # 現在時刻の取得
        now = datetime.now()

        # "0915"のように，現在の日付を文字列で取得
        nowStr = now.strftime("%m%d")

        # 画像があるディレクトリ一覧
        datalist = os.listdir(data_path)

        for dataDate in datalist:
            if nowStr != dataDate:

                # 圧縮対象のディレクトリのPATH
                targetDir = os.path.join(data_path, dataDate)
                # 圧縮先のzipファイルのPATH名
                saveZip = os.path.join(zip_path, dataDate)

                # その日の分を圧縮させてディレクトリを消す
                shutil.make_archive(saveZip, 'zip', root_dir=targetDir)
                shutil.rmtree(os.path.join(data_path, dataDate))

                print("--- remove the directory ---")
                print(dataDate)

                # 新しい日付のディレクトリを作成
                os.mkdir(os.path.join(data_path, nowStr))

                print("--- New Directory ---")
                print(nowStr)
