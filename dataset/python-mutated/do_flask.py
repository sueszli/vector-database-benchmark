from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if False:
        return 10
    return '<h1>Home</h1>'

@app.route('/signin', methods=['GET'])
def signin_form():
    if False:
        return 10
    return '<form action="/signin" method="post">\n              <p><input name="username"></p>\n              <p><input name="password" type="password"></p>\n              <p><button type="submit">Sign In</button></p>\n              </form>'

@app.route('/signin', methods=['POST'])
def signin():
    if False:
        while True:
            i = 10
    if request.form['username'] == 'admin' and request.form['password'] == 'password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'
if __name__ == '__main__':
    app.run()