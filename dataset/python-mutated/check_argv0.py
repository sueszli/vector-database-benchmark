from __future__ import annotations
import sys
from pathlib import Path

def main() -> int:
    if False:
        return 10
    path = Path(sys.argv[0])
    if sys.argv[1] == 'absolute':
        if not path.is_absolute():
            raise RuntimeError(f'sys.argv[0] is not an absolute path: {path}')
        if not path.exists():
            raise RuntimeError(f'sys.argv[0] does not exist: {path}')
    elif path.is_absolute():
        raise RuntimeError(f'sys.argv[0] is an absolute path: {path}')
    return 0
if __name__ == '__main__':
    raise sys.exit(main())