"""A simple command line tool for testing purposes."""
import fire

def identity(arg=None):
    if False:
        i = 10
        return i + 15
    return (arg, type(arg))

def main(_=None):
    if False:
        i = 10
        return i + 15
    fire.Fire(identity, name='identity')
if __name__ == '__main__':
    main()