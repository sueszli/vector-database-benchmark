from __future__ import annotations
from flask import Flask
from flask import Response
app = Flask(__name__)

@app.after_request
def after_sync(response: Response) -> Response:
    if False:
        for i in range(10):
            print('nop')
    return Response()

@app.after_request
async def after_async(response: Response) -> Response:
    return Response()

@app.before_request
def before_sync() -> None:
    if False:
        while True:
            i = 10
    ...

@app.before_request
async def before_async() -> None:
    ...

@app.teardown_appcontext
def teardown_sync(exc: BaseException | None) -> None:
    if False:
        print('Hello World!')
    ...

@app.teardown_appcontext
async def teardown_async(exc: BaseException | None) -> None:
    ...