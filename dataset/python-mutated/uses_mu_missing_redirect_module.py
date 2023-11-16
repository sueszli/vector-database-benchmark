from __future__ import annotations
from ..module_utils import missing_redirect_target_module

def main():
    if False:
        i = 10
        return i + 15
    raise Exception('should never get here')
if __name__ == '__main__':
    main()