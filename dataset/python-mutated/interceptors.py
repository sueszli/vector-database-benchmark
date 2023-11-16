"""Infrastructure for intercepting requests."""
import enum
import dataclasses
from typing import Callable, List, Optional
from qutebrowser.qt.core import QUrl

class ResourceType(enum.Enum):
    """Possible request types that can be received.

    Currently corresponds to the QWebEngineUrlRequestInfo Enum:
    https://doc.qt.io/qt-6/qwebengineurlrequestinfo.html#ResourceType-enum
    """
    main_frame = 0
    sub_frame = 1
    stylesheet = 2
    script = 3
    image = 4
    font_resource = 5
    sub_resource = 6
    object = 7
    media = 8
    worker = 9
    shared_worker = 10
    prefetch = 11
    favicon = 12
    xhr = 13
    ping = 14
    service_worker = 15
    csp_report = 16
    plugin_resource = 17
    preload_main_frame = 19
    preload_sub_frame = 20
    websocket = 254
    unknown = 255

class RedirectException(Exception):
    """Raised when the request was invalid, or a request was already made."""

@dataclasses.dataclass
class Request:
    """A request which can be intercepted/blocked."""
    first_party_url: Optional[QUrl]
    request_url: QUrl
    is_blocked: bool = False
    resource_type: Optional[ResourceType] = None

    def block(self) -> None:
        if False:
            return 10
        'Block this request.'
        self.is_blocked = True

    def redirect(self, url: QUrl, *, ignore_unsupported: bool=False) -> None:
        if False:
            print('Hello World!')
        "Redirect this request.\n\n        Only some types of requests can be successfully redirected.\n        Improper use of this method can result in redirect loops.\n\n        This method will throw a RedirectException if the request was not possible.\n\n        Args:\n            url: The QUrl to try to redirect to.\n            ignore_unsupported: If set to True, request methods which can't be\n                redirected (such as POST) are silently ignored instead of throwing an\n                exception.\n        "
        raise NotImplementedError
InterceptorType = Callable[[Request], None]
_interceptors: List[InterceptorType] = []

def register(interceptor: InterceptorType) -> None:
    if False:
        return 10
    _interceptors.append(interceptor)

def run(info: Request) -> None:
    if False:
        while True:
            i = 10
    for interceptor in _interceptors:
        interceptor(info)