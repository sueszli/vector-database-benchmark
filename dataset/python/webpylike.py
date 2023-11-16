"""Implements web.py like dispatching. What this module does not
implement is a stream system that hooks into sys.stdout like web.py
provides.
"""
import re

from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import MethodNotAllowed
from werkzeug.exceptions import NotFound
from werkzeug.exceptions import NotImplemented
from werkzeug.wrappers import Request
from werkzeug.wrappers import Response  # noqa: F401


class View:
    """Baseclass for our views."""

    def __init__(self, app, req):
        self.app = app
        self.req = req

    def GET(self):
        raise MethodNotAllowed()

    POST = DELETE = PUT = GET

    def HEAD(self):
        return self.GET()


class WebPyApp:
    """
    An interface to a web.py like application.  It works like the web.run
    function in web.py
    """

    def __init__(self, urls, views):
        self.urls = [
            (re.compile(f"^{urls[i]}$"), urls[i + 1]) for i in range(0, len(urls), 2)
        ]
        self.views = views

    def __call__(self, environ, start_response):
        try:
            req = Request(environ)
            for regex, view in self.urls:
                match = regex.match(req.path)
                if match is not None:
                    view = self.views[view](self, req)
                    if req.method not in ("GET", "HEAD", "POST", "DELETE", "PUT"):
                        raise NotImplemented()  # noqa: F901
                    resp = getattr(view, req.method)(*match.groups())
                    break
            else:
                raise NotFound()
        except HTTPException as e:
            resp = e
        return resp(environ, start_response)
