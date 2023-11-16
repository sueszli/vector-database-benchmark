from pathlib import Path
from typing import Callable, Optional, Union
import fastapi
from .page import page as ui_page

class APIRouter(fastapi.APIRouter):

    def page(self, path: str, *, title: Optional[str]=None, viewport: Optional[str]=None, favicon: Optional[Union[str, Path]]=None, dark: Optional[bool]=..., response_timeout: float=3.0, **kwargs) -> Callable:
        if False:
            for i in range(10):
                print('nop')
        "Page\n\n        Creates a new page at the given route.\n        Each user will see a new instance of the page.\n        This means it is private to the user and not shared with others\n        (as it is done `when placing elements outside of a page decorator <https://nicegui.io/documentation#auto-index_page>`_).\n\n        :param path: route of the new page (path must start with '/')\n        :param title: optional page title\n        :param viewport: optional viewport meta tag content\n        :param favicon: optional relative filepath or absolute URL to a favicon (default: `None`, NiceGUI icon will be used)\n        :param dark: whether to use Quasar's dark mode (defaults to `dark` argument of `run` command)\n        :param response_timeout: maximum time for the decorated function to build the page (default: 3.0)\n        :param kwargs: additional keyword arguments passed to FastAPI's @app.get method\n        "
        return ui_page(path, title=title, viewport=viewport, favicon=favicon, dark=dark, response_timeout=response_timeout, api_router=self, **kwargs)