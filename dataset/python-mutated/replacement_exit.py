import os
import sys
from scalene.scalene_profiler import Scalene

@Scalene.shim
def replacement_exit(scalene: Scalene) -> None:
    if False:
        return 10
    '\n    Shims out the unconditional exit with\n    the "neat exit" (which raises the SystemExit error and\n    allows Scalene to exit neatly)\n    '
    os._exit = sys.exit