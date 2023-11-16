from flask import Flask
app = Flask(__name__)

@app.route('/')
def main():
    if False:
        while True:
            i = 10
    return 'Welcome!'
if __name__ == '__main__':
    pass