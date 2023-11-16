from flask import Flask, render_template
from flask import Flask, send_file, make_response, send_from_directory
import pandas as pd
import os
from datetime import datetime
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)
Host_name = 'localhost'
zip_path = '/home/pi/Desktop/pi_system/raspi/zipfile/'

@app.route('/collect')
def collect():
    if False:
        print('Hello World!')
    fileList = os.listdir(zip_path)
    now = datetime.now()
    nowStr = now.strftime('%m%d')
    downloadFile = zip_path + nowStr + '.zip'
    downloadFileName = nowStr + '.zip'
    return send_file(downloadFile, mimetype='application/zip', as_attachment=True, attachment_filename=downloadFileName)
if __name__ == '__main__':
    print(app.url_map)
    app.run(host='localhost', port=3000)