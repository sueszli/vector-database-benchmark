import sys

def fail(msg):
    if False:
        print('Hello World!')
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()
    sys.exit(1)

def main():
    if False:
        while True:
            i = 10
    fail('This installation method has been deprecated. See https://github.com/pypa/pipx for current installation instructions.')
if __name__ == '__main__':
    main()