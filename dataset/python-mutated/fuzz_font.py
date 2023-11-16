import atheris
with atheris.instrument_imports():
    import sys
    import fuzzers

def TestOneInput(data):
    if False:
        i = 10
        return i + 15
    try:
        fuzzers.fuzz_font(data)
    except Exception:
        pass

def main():
    if False:
        while True:
            i = 10
    fuzzers.enable_decompressionbomb_error()
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
    fuzzers.disable_decompressionbomb_error()
if __name__ == '__main__':
    main()