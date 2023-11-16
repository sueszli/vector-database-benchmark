import atheris
with atheris.instrument_imports():
    import sys
    import fuzzers

def TestOneInput(data):
    if False:
        return 10
    try:
        fuzzers.fuzz_image(data)
    except Exception:
        pass

def main():
    if False:
        print('Hello World!')
    fuzzers.enable_decompressionbomb_error()
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
    fuzzers.disable_decompressionbomb_error()
if __name__ == '__main__':
    main()