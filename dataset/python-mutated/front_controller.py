"""
@author: Gordeev Andrey <gordeev.and.and@gmail.com>

*TL;DR
Provides a centralized entry point that controls and manages request handling.
"""
from __future__ import annotations
from typing import Any

class MobileView:

    def show_index_page(self) -> None:
        if False:
            while True:
                i = 10
        print('Displaying mobile index page')

class TabletView:

    def show_index_page(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        print('Displaying tablet index page')

class Dispatcher:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.mobile_view = MobileView()
        self.tablet_view = TabletView()

    def dispatch(self, request: Request) -> None:
        if False:
            while True:
                i = 10
        '\n        This function is used to dispatch the request based on the type of device.\n        If it is a mobile, then mobile view will be called and if it is a tablet,\n        then tablet view will be called.\n        Otherwise, an error message will be printed saying that cannot dispatch the request.\n        '
        if request.type == Request.mobile_type:
            self.mobile_view.show_index_page()
        elif request.type == Request.tablet_type:
            self.tablet_view.show_index_page()
        else:
            print('Cannot dispatch the request')

class RequestController:
    """front controller"""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dispatcher = Dispatcher()

    def dispatch_request(self, request: Any) -> None:
        if False:
            print('Hello World!')
        '\n        This function takes a request object and sends it to the dispatcher.\n        '
        if isinstance(request, Request):
            self.dispatcher.dispatch(request)
        else:
            print('request must be a Request object')

class Request:
    """request"""
    mobile_type = 'mobile'
    tablet_type = 'tablet'

    def __init__(self, request):
        if False:
            print('Hello World!')
        self.type = None
        request = request.lower()
        if request == self.mobile_type:
            self.type = self.mobile_type
        elif request == self.tablet_type:
            self.type = self.tablet_type

def main():
    if False:
        while True:
            i = 10
    "\n    >>> front_controller = RequestController()\n\n    >>> front_controller.dispatch_request(Request('mobile'))\n    Displaying mobile index page\n\n    >>> front_controller.dispatch_request(Request('tablet'))\n    Displaying tablet index page\n\n    >>> front_controller.dispatch_request(Request('desktop'))\n    Cannot dispatch the request\n\n    >>> front_controller.dispatch_request('mobile')\n    request must be a Request object\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()