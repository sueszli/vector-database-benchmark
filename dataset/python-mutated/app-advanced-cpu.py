import os
import sys
from crowdcounting import CrowdCountModelPose, CrowdCountModelMCNN, Router
from werkzeug.utils import secure_filename
from flask import Flask, Response, json, jsonify, render_template, request, send_file, send_from_directory
import numpy as np
import logging
import time
import argparse
import cv2
import urllib
import base64
from io import BytesIO
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
parser = argparse.ArgumentParser(description='A demo app.')
parser.add_argument('-p', '--path', help='Path to MCNN model file', required=True)
args = parser.parse_args()
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'images/')
image_names = os.listdir(target)
image_names.sort()
actual_dict = {'1.jpg': 3, '2.jpg': 60, '3.jpg': 7}
actual_counts = ['Actual count: ' + str(actual_dict[image_name]) for image_name in image_names]
gpu_id = -1
mcnn_model_path = args.path
model = CrowdCountModelPose(gpu_id)

@app.route('/', methods=['GET'])
def load():
    if False:
        while True:
            i = 10
    return render_template('index.html', image_names=image_names, actual_counts=actual_counts)

@app.route('/upload/<filename>')
def send_image(filename):
    if False:
        i = 10
        return i + 15
    return send_from_directory('images', filename)

@app.route('/uploadfile', methods=['POST'])
def use_upload_file():
    if False:
        while True:
            i = 10
    uploaded_file = request.files['file']
    request_data = uploaded_file.read()
    result = model.score(request_data, return_image=True, img_dim=1750)
    pred = result['pred']
    scored_image = result['image']
    txt = 'Predicted count: {0}'.format(pred)
    logger.info('use uploaded file')
    return render_template('result.html', scored_image=scored_image, txt=txt)

@app.route('/sitefile', methods=['POST'])
def use_site_file():
    if False:
        print('Hello World!')
    target = os.path.join(APP_ROOT, 'images')
    result = request.form['fileindex']
    local_image = '/'.join([target, result])
    local_image = secure_filename(local_image)
    with open(local_image, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=1750)
    pred = result['pred']
    scored_image = result['image']
    txt = 'Predicted count: {0}'.format(pred)
    return render_template('result.html', scored_image=scored_image, txt=txt)

@app.route('/score', methods=['POST'])
def score():
    if False:
        for i in range(10):
            print('nop')
    result = model.score(request.data, return_image=False, img_dim=1750)
    js = json.dumps({'count': int(np.round(result['pred']))})
    resp = Response(js, status=200, mimetype='application/json')
    return resp

@app.route('/score_alt', methods=['POST'])
def score_alt():
    if False:
        return 10
    result = model.score(request.data, return_image=True, img_dim=1750)
    t = urllib.parse.unquote(result['image'])
    image = base64.b64decode(t)
    return send_file(BytesIO(image), as_attachment=True, attachment_filename='pred.png', mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)