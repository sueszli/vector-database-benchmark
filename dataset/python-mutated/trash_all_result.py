from typing import NamedTuple, List

class TrashAllResult(NamedTuple('TrashAllResult', [('failed_paths', List[str])])):

    def any_failure(self):
        if False:
            print('Hello World!')
        return len(self.failed_paths) > 0