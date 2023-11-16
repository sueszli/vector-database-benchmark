from typing import Optional, Dict, Any
from .PaginationLinks import PaginationLinks
from .PaginationMetadata import PaginationMetadata
from .ResponseMeta import ResponseMeta

class PaginationManager:

    def __init__(self, limit: int) -> None:
        if False:
            i = 10
            return i + 15
        self.limit = limit
        self.meta = None
        self.links = None

    def setResponseMeta(self, meta: Optional[Dict[str, Any]]) -> None:
        if False:
            print('Hello World!')
        self.meta = None
        if meta:
            page = None
            if 'page' in meta:
                page = PaginationMetadata(**meta['page'])
            self.meta = ResponseMeta(page)

    def setLinks(self, links: Optional[Dict[str, str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.links = PaginationLinks(**links) if links else None

    def setLimit(self, new_limit: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the limit of items per page.\n\n        :param new_limit: The new limit of items per page\n        '
        self.limit = new_limit
        self.reset()

    def reset(self) -> None:
        if False:
            return 10
        '\n        Sets the metadata and links to None.\n        '
        self.meta = None
        self.links = None