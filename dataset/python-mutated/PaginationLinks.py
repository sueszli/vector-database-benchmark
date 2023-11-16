from typing import Optional

class PaginationLinks:
    """Model containing pagination links."""

    def __init__(self, first: Optional[str]=None, last: Optional[str]=None, next: Optional[str]=None, prev: Optional[str]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new digital factory project response object\n        :param first: The URL for the first page.\n        :param last: The URL for the last page.\n        :param next: The URL for the next page.\n        :param prev: The URL for the prev page.\n        :param kwargs:\n        '
        self.first_page = first
        self.last_page = last
        self.next_page = next
        self.prev_page = prev

    def __str__(self) -> str:
        if False:
            return 10
        return 'Pagination Links | First: {}, Last: {}, Next: {}, Prev: {}'.format(self.first_page, self.last_page, self.next_page, self.prev_page)