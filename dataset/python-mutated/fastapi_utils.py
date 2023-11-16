"""Collection of utilities for FastAPI apps."""
import inspect
from typing import Any, Type
from fastapi import FastAPI, Form
from pydantic import BaseModel

def as_form(cls: Type[BaseModel]) -> Any:
    if False:
        print('Hello World!')
    'Adds an as_form class method to decorated models.\n\n    The as_form class method can be used with FastAPI endpoints\n    '
    new_params = [inspect.Parameter(field.alias, inspect.Parameter.POSITIONAL_ONLY, default=Form(field.default) if not field.required else Form(...)) for field in cls.__fields__.values()]

    async def _as_form(**data):
        return cls(**data)
    sig = inspect.signature(_as_form)
    sig = sig.replace(parameters=new_params)
    _as_form.__signature__ = sig
    setattr(cls, 'as_form', _as_form)
    return cls

def patch_fastapi(app: FastAPI) -> None:
    if False:
        while True:
            i = 10
    'Patch function to allow relative url resolution.\n\n    This patch is required to make fastapi fully functional with a relative url path.\n    This code snippet can be copy-pasted to any Fastapi application.\n    '
    from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
    from starlette.requests import Request
    from starlette.responses import HTMLResponse

    async def redoc_ui_html(req: Request) -> HTMLResponse:
        assert app.openapi_url is not None
        redoc_ui = get_redoc_html(openapi_url='./' + app.openapi_url.lstrip('/'), title=app.title + ' - Redoc UI')
        return HTMLResponse(redoc_ui.body.decode('utf-8'))

    async def swagger_ui_html(req: Request) -> HTMLResponse:
        assert app.openapi_url is not None
        swagger_ui = get_swagger_ui_html(openapi_url='./' + app.openapi_url.lstrip('/'), title=app.title + ' - Swagger UI', oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url)
        request_interceptor = 'requestInterceptor: (e)  => {\n\t\t\tvar url = window.location.origin + window.location.pathname\n\t\t\turl = url.substring( 0, url.lastIndexOf( "/" ) + 1);\n\t\t\turl = e.url.replace(/http(s)?:\\/\\/[^/]*\\//i, url);\n\t\t\te.contextUrl = url\n\t\t\te.url = url\n\t\t\treturn e;}'
        return HTMLResponse(swagger_ui.body.decode('utf-8').replace("dom_id: '#swagger-ui',", "dom_id: '#swagger-ui',\n\t\t" + request_interceptor + ','))
    routes_new = []
    for app_route in app.routes:
        if app_route.path == '/docs':
            continue
        if app_route.path == '/redoc':
            continue
        routes_new.append(app_route)
    app.router.routes = routes_new
    assert app.docs_url is not None
    app.add_route(app.docs_url, swagger_ui_html, include_in_schema=False)
    assert app.redoc_url is not None
    app.add_route(app.redoc_url, redoc_ui_html, include_in_schema=False)
    from starlette import graphql
    graphql.GRAPHIQL = graphql.GRAPHIQL.replace('({{REQUEST_PATH}}', '("." + {{REQUEST_PATH}}')