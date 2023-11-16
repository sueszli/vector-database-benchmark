from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    if False:
        print('Hello World!')
    return 'Hello World!'