import sys
import atheris
with atheris.instrument_imports():
    from websockets.exceptions import SecurityError
    from websockets.http11 import Response
    from websockets.streams import StreamReader

def test_one_input(data):
    if False:
        i = 10
        return i + 15
    reader = StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    parser = Response.parse(reader.read_line, reader.read_exact, reader.read_to_eof)
    try:
        next(parser)
    except StopIteration as exc:
        assert isinstance(exc.value, Response)
        return
    except (EOFError, SecurityError, LookupError, ValueError):
        return
    raise RuntimeError("parsing didn't complete")

def main():
    if False:
        while True:
            i = 10
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
if __name__ == '__main__':
    main()