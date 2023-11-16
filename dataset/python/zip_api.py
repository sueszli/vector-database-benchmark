#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Flask を使用した画像取得プログラム変更 9/2
# RaspberryPi に API として呼び出し
# http で呼び出して画像を撮って保存させる
# 夜に取得した画像を一日分圧縮してサーバに保存する

from flask import Flask, render_template
from flask import Flask, send_file, make_response, send_from_directory
import pandas as pd
import os

from datetime import datetime
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# localhostは変更してください
Host_name = "localhost"
zip_path = '/home/pi/Desktop/pi_system/raspi/zipfile/'

@app.route("/collect")
def collect():

    fileList = os.listdir(zip_path)

    now = datetime.now()
    nowStr = now.strftime("%m%d")

    # ここのPATH指定は，なんとかしてください
    downloadFile = zip_path + nowStr + ".zip"
    downloadFileName = nowStr + ".zip"

    return send_file(downloadFile, mimetype="application/zip", as_attachment=True, attachment_filename=downloadFileName)

if __name__ == "__main__":
    print(app.url_map)
    app.run(host="localhost", port=3000)
