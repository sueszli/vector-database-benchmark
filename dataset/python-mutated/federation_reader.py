import sys
from synapse.app.generic_worker import start
from synapse.util.logcontext import LoggingContext

def main() -> None:
    if False:
        i = 10
        return i + 15
    with LoggingContext('main'):
        start(sys.argv[1:])
if __name__ == '__main__':
    main()