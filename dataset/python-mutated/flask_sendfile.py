import io
from flask import Flask, send_file
app = Flask(__name__)

@app.route('/')
def index():
    if False:
        while True:
            i = 10
    buf = io.BytesIO()
    buf.write(b'hello world')
    buf.seek(0)
    return send_file(buf, attachment_filename='testing.txt', as_attachment=True)