import sys
from streamlit import util

class modified_sys_path:
    """A context for prepending a directory to sys.path for a second."""

    def __init__(self, main_script_path: str):
        if False:
            for i in range(10):
                print('nop')
        self._main_script_path = main_script_path
        self._added_path = False

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return util.repr_(self)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if self._main_script_path not in sys.path:
            sys.path.insert(0, self._main_script_path)
            self._added_path = True

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self._added_path:
            try:
                sys.path.remove(self._main_script_path)
            except ValueError:
                pass
        return False