import sys

def main():
    if False:
        return 10
    args = sys.argv[1:]
    for filename in args:
        process(filename)

def process(filename):
    if False:
        for i in range(10):
            print('nop')
    try:
        f = open(filename, 'r')
    except IOError as msg:
        sys.stderr.write("%s: can't open: %s\n" % (filename, str(msg)))
        return
    with f:
        data = f.read()
    if data[:2] != '/*':
        sys.stderr.write('%s does not begin with C comment\n' % filename)
        return
    try:
        f = open(filename, 'w')
    except IOError as msg:
        sys.stderr.write("%s: can't write: %s\n" % (filename, str(msg)))
        return
    with f:
        sys.stderr.write('Processing %s ...\n' % filename)
        magic = 'Py_'
        for c in filename:
            if ord(c) <= 128 and c.isalnum():
                magic = magic + c.upper()
            else:
                magic = magic + '_'
        print('#ifndef', magic, file=f)
        print('#define', magic, file=f)
        print('#ifdef __cplusplus', file=f)
        print('extern "C" {', file=f)
        print('#endif', file=f)
        print(file=f)
        f.write(data)
        print(file=f)
        print('#ifdef __cplusplus', file=f)
        print('}', file=f)
        print('#endif', file=f)
        print('#endif /*', '!' + magic, '*/', file=f)
if __name__ == '__main__':
    main()