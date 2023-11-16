"""This is a simple web service/HTTP wrapper for OCRmyPDF.

This may be more convenient than the command line tool for some Docker users.
Note that OCRmyPDF uses Ghostscript, which is licensed under AGPLv3+. While
OCRmyPDF is under GPLv3, this file is distributed under the Affero GPLv3+ license,
to emphasize that SaaS deployments should make sure they comply with
Ghostscript's license as well as OCRmyPDF's.
"""
from __future__ import annotations
import os
import shlex
from subprocess import run
from tempfile import TemporaryDirectory
from flask import Flask, Response, request, send_from_directory
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.secret_key = 'secret'
app.config['MAX_CONTENT_LENGTH'] = 50000000
app.config.from_envvar('OCRMYPDF_WEBSERVICE_SETTINGS', silent=True)
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    if False:
        print('Hello World!')
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def do_ocrmypdf(file):
    if False:
        print('Hello World!')
    uploaddir = TemporaryDirectory(prefix='ocrmypdf-upload')
    downloaddir = TemporaryDirectory(prefix='ocrmypdf-download')
    filename = secure_filename(file.filename)
    up_file = os.path.join(uploaddir.name, filename)
    file.save(up_file)
    down_file = os.path.join(downloaddir.name, filename)
    cmd_args = [arg for arg in shlex.split(request.form['params'])]
    if '--sidecar' in cmd_args:
        return Response('--sidecar not supported', 501, mimetype='text/plain')
    ocrmypdf_args = ['ocrmypdf', *cmd_args, up_file, down_file]
    proc = run(ocrmypdf_args, capture_output=True, encoding='utf-8', check=False)
    if proc.returncode != 0:
        stderr = proc.stderr
        return Response(stderr, 400, mimetype='text/plain')
    return send_from_directory(downloaddir.name, filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if False:
        print('Hello World!')
    if request.method == 'POST':
        if 'file' not in request.files:
            return Response('No file in POST', 400, mimetype='text/plain')
        file = request.files['file']
        if file.filename == '':
            return Response('Empty filename', 400, mimetype='text/plain')
        if not allowed_file(file.filename):
            return Response('Invalid filename', 400, mimetype='text/plain')
        if file and allowed_file(file.filename):
            return do_ocrmypdf(file)
        return Response('Some other problem', 400, mimetype='text/plain')
    return '\n    <!doctype html>\n    <title>OCRmyPDF webservice</title>\n    <h1>Upload a PDF (debug UI)</h1>\n    <form method=post enctype=multipart/form-data>\n      <label for="args">Command line parameters</label>\n      <input type=textbox name=params>\n      <label for="file">File to upload</label>\n      <input type=file name=file>\n      <input type=submit value=Upload>\n    </form>\n    <h4>Notice</h2>\n    <div style="font-size: 70%; max-width: 34em;">\n    <p>This is a webservice wrapper for OCRmyPDF.</p>\n    <p>Copyright 2019 James R. Barlow</p>\n    <p>This program is free software: you can redistribute it and/or modify\n    it under the terms of the GNU Affero General Public License as published by\n    the Free Software Foundation, either version 3 of the License, or\n    (at your option) any later version.\n    </p>\n    <p>This program is distributed in the hope that it will be useful,\n    but WITHOUT ANY WARRANTY; without even the implied warranty of\n    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n    GNU General Public License for more details.\n    </p>\n    <p>\n    You should have received a copy of the GNU Affero General Public License\n    along with this program.  If not, see &lt;http://www.gnu.org/licenses/&gt;.\n    </p>\n    </div>\n    '
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)