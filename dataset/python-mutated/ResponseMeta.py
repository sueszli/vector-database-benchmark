from typing import Optional
from .PaginationMetadata import PaginationMetadata

class ResponseMeta:
    """Class representing the metadata included in a Digital Library response (if any)"""

    def __init__(self, page: Optional[PaginationMetadata]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Creates a new digital factory project response object\n        :param page: Metadata related to pagination\n        :param kwargs:\n        '
        self.page = page
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Response Meta | {}'.format(self.page)