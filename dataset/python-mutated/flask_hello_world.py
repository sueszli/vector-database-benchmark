from flask import Flask
app = Flask(__name__)

@app.route('/hello')
def hello():
    if False:
        i = 10
        return i + 15
    return 'Hello World!'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)