"""Example for aiohttp.web basic server with cookies."""
from pprint import pformat
from typing import NoReturn
from aiohttp import web
tmpl = '<html>\n    <body>\n        <a href="/login">Login</a><br/>\n        <a href="/logout">Logout</a><br/>\n        <pre>{}</pre>\n    </body>\n</html>'

async def root(request: web.Request) -> web.StreamResponse:
    resp = web.Response(content_type='text/html')
    resp.text = tmpl.format(pformat(request.cookies))
    return resp

async def login(request: web.Request) -> NoReturn:
    exc = web.HTTPFound(location='/')
    exc.set_cookie('AUTH', 'secret')
    raise exc

async def logout(request: web.Request) -> NoReturn:
    exc = web.HTTPFound(location='/')
    exc.del_cookie('AUTH')
    raise exc

def init() -> web.Application:
    if False:
        for i in range(10):
            print('nop')
    app = web.Application()
    app.router.add_get('/', root)
    app.router.add_get('/login', login)
    app.router.add_get('/logout', logout)
    return app
web.run_app(init())