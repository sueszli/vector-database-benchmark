from __future__ import annotations
from http import HTTPStatus
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import NotFound
from flask import Flask
app = Flask(__name__)

@app.errorhandler(400)
@app.errorhandler(HTTPStatus.BAD_REQUEST)
@app.errorhandler(BadRequest)
def handle_400(e: BadRequest) -> str:
    if False:
        i = 10
        return i + 15
    return ''

@app.errorhandler(ValueError)
def handle_custom(e: ValueError) -> str:
    if False:
        while True:
            i = 10
    return ''

@app.errorhandler(ValueError)
def handle_accept_base(e: Exception) -> str:
    if False:
        print('Hello World!')
    return ''

@app.errorhandler(BadRequest)
@app.errorhandler(404)
def handle_multiple(e: BadRequest | NotFound) -> str:
    if False:
        while True:
            i = 10
    return ''