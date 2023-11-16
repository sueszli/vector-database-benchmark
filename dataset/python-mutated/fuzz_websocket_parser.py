import sys
import atheris
with atheris.instrument_imports():
    from websockets.exceptions import PayloadTooBig, ProtocolError
    from websockets.frames import Frame
    from websockets.streams import StreamReader

def test_one_input(data):
    if False:
        while True:
            i = 10
    fdp = atheris.FuzzedDataProvider(data)
    mask = fdp.ConsumeBool()
    max_size_enabled = fdp.ConsumeBool()
    max_size = fdp.ConsumeInt(4)
    payload = fdp.ConsumeBytes(atheris.ALL_REMAINING)
    reader = StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    parser = Frame.parse(reader.read_exact, mask=mask, max_size=max_size if max_size_enabled else None)
    try:
        next(parser)
    except StopIteration as exc:
        assert isinstance(exc.value, Frame)
        return
    except (EOFError, UnicodeDecodeError, PayloadTooBig, ProtocolError):
        return
    raise RuntimeError("parsing didn't complete")

def main():
    if False:
        print('Hello World!')
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
if __name__ == '__main__':
    main()