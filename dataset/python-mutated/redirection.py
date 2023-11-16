"""
This file is part of Commix Project (https://commixproject.com).
Copyright (c) 2014-2023 Anastasios Stasinopoulos (@ancst).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For more see the file 'readme/COPYING' for copying permission.
"""
import re
import sys
import errno
import base64
try:
    from base64 import encodebytes
except ImportError:
    from base64 import encodestring as encodebytes
from src.utils import menu
from src.utils import settings
from src.utils import common
from src.core.requests import requests
from socket import error as SocketError
from src.core.injections.controller import checks
from src.thirdparty.six.moves import input as _input
from src.thirdparty.six.moves import urllib as _urllib
from src.thirdparty.six.moves import http_client as _http_client
from src.thirdparty.colorama import Fore, Back, Style, init

def do_check(request, url, redirect_url):
    if False:
        print('Hello World!')
    "\n  This functinality is based on Filippo's Valsorda script [1].\n  ---\n  [1] https://gist.github.com/FiloSottile/2077115\n  "

    class Request(_urllib.request.Request):

        def get_method(self):
            if False:
                print('Hello World!')
            return settings.HTTPMETHOD.HEAD

    class RedirectHandler(_urllib.request.HTTPRedirectHandler, object):
        """
    Subclass the HTTPRedirectHandler to make it use our
    Request also on the redirected URL
    """

        def redirect_request(self, request, fp, code, msg, headers, newurl):
            if False:
                while True:
                    i = 10
            if code in (301, 302, 303, 307):
                settings.REDIRECT_CODE = code
                return Request(newurl.replace(' ', '%20'), data=request.data, headers=request.headers)
            else:
                err_msg = str(_urllib.error.HTTPError(request.get_full_url(), code, msg, headers, fp)).replace(': ', ' (')
                print(settings.print_critical_msg(err_msg + ').'))
                raise SystemExit()
    try:
        opener = _urllib.request.build_opener(RedirectHandler())
        _urllib.request.install_opener(opener)
        response = _urllib.request.urlopen(request, timeout=settings.TIMEOUT)
    except (SocketError, _urllib.error.HTTPError, _urllib.error.URLError, _http_client.BadStatusLine, _http_client.IncompleteRead, _http_client.InvalidURL) as err_msg:
        requests.crawler_request(redirect_url)
    try:
        if settings.CRAWLING and redirect_url in settings.HREF_SKIPPED:
            return redirect_url
        elif settings.CRAWLING and url in settings.HREF_SKIPPED:
            return url
        else:
            while True:
                if not settings.FOLLOW_REDIRECT:
                    if settings.CRAWLED_URLS_NUM != 0 and settings.CRAWLED_SKIPPED_URLS_NUM != 0:
                        print(settings.SINGLE_WHITESPACE)
                message = 'Got a ' + str(settings.REDIRECT_CODE) + " redirect to '" + redirect_url
                message += "'. Do you want to follow? [Y/n] > "
                redirection_option = common.read_input(message, default='Y', check_batch=True)
                if redirection_option in settings.CHOICE_YES:
                    settings.FOLLOW_REDIRECT = True
                    info_msg = "Following redirection to '" + redirect_url + "'. "
                    print(settings.print_info_msg(info_msg))
                    if settings.CRAWLING:
                        settings.HREF_SKIPPED.append(url)
                    return checks.check_http_s(redirect_url)
                elif redirection_option in settings.CHOICE_NO:
                    settings.FOLLOW_REDIRECT = False
                    if settings.CRAWLING:
                        settings.HREF_SKIPPED.append(url)
                    return url
                elif redirection_option in settings.CHOICE_QUIT:
                    raise SystemExit()
                else:
                    common.invalid_option(redirection_option)
                    pass
    except AttributeError:
        return url