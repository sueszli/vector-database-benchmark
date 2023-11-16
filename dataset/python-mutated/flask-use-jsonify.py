import flask
import json
app = flask.Flask(__name__)

@app.route('/user')
def user():
    if False:
        while True:
            i = 10
    user_dict = get_user(request.args.get('id'))
    return json.dumps(user_dict)
from json import dumps

@app.route('/user')
def user():
    if False:
        for i in range(10):
            print('nop')
    user_dict = get_user(request.args.get('id'))
    return dumps(user_dict)

def dumps():
    if False:
        return 10
    pass

def test_empty_dumps():
    if False:
        i = 10
        return i + 15
    dumps()