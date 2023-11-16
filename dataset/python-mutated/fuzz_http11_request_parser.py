import sys
import atheris
with atheris.instrument_imports():
    from websockets.exceptions import SecurityError
    from websockets.http11 import Request
    from websockets.streams import StreamReader

def test_one_input(data):
    if False:
        while True:
            i = 10
    reader = StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    parser = Request.parse(reader.read_line)
    try:
        next(parser)
    except StopIteration as exc:
        assert isinstance(exc.value, Request)
        return
    except (EOFError, SecurityError, ValueError):
        return
    raise RuntimeError("parsing didn't complete")

def main():
    if False:
        for i in range(10):
            print('nop')
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
if __name__ == '__main__':
    main()