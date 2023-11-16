""" Provide a request handler that returns a page displaying a document.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from tornado.web import StaticFileHandler
from bokeh.settings import settings
__all__ = ('StaticHandler',)

class StaticHandler(StaticFileHandler):
    """ Implements a custom Tornado static file handler for BokehJS
    JavaScript and CSS resources.

    """

    def __init__(self, tornado_app, *args, **kw) -> None:
        if False:
            return 10
        kw['path'] = settings.bokehjs_path()
        super().__init__(tornado_app, *args, **kw)

    @classmethod
    def append_version(cls, path: str) -> str:
        if False:
            i = 10
            return i + 15
        if settings.dev:
            return path
        else:
            version = StaticFileHandler.get_version(dict(static_path=settings.bokehjs_path()), path)
            return f'{path}?v={version}'