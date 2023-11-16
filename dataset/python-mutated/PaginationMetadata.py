from typing import Optional

class PaginationMetadata:
    """Class representing the metadata related to pagination."""

    def __init__(self, total_count: Optional[int]=None, total_pages: Optional[int]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new digital factory project response object\n        :param total_count: The total count of items.\n        :param total_pages: The total number of pages when pagination is applied.\n        :param kwargs:\n        '
        self.total_count = total_count
        self.total_pages = total_pages
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'PaginationMetadata | Total Count: {}, Total Pages: {}'.format(self.total_count, self.total_pages)