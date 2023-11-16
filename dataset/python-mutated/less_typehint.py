from flask import send_file
app = Flask(__name__)

def download_not_flask_route(filename):
    if False:
        i = 10
        return i + 15
    return send_file(filename)

@app.route('/<path:filename>')
def download_file(filename: str):
    if False:
        i = 10
        return i + 15
    return send_file(filename)