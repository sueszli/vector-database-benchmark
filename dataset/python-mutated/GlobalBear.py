from coalib.bears.Bear import Bear
from coalib.bears.BEAR_KIND import BEAR_KIND

class GlobalBear(Bear):
    """
    A GlobalBear analyzes semantic facts across several files.

    The results of a GlobalBear will be presented grouped by the origin Bear.
    Therefore Results spanning across multiple files are allowed and will be
    handled correctly.

    If you are inspecting a single file at a time, you should consider
    using a LocalBear.
    """

    def __init__(self, file_dict, section, message_queue, timeout=0):
        if False:
            while True:
                i = 10
        '\n        Constructs a new GlobalBear.\n\n        :param file_dict: The dictionary of {filename: file contents}.\n\n        See :class:`coalib.bears.Bear` for other parameters.\n        '
        Bear.__init__(self, section, message_queue, timeout)
        self.file_dict = file_dict

    @staticmethod
    def kind():
        if False:
            for i in range(10):
                print('nop')
        return BEAR_KIND.GLOBAL

    def run(self, *args, dependency_results=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Handles all files in file_dict.\n\n        :param dependency_results: The dictionary of {bear name:\n                                   result list}.\n        :return: A list of Result type.\n\n        See :class:`coalib.bears.Bear` for `run` method description.\n        '
        raise NotImplementedError('This function has to be implemented for a runnable bear.')