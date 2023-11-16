import logging

def foo(s):
    if False:
        i = 10
        return i + 15
    return 10 / int(s)

def bar(s):
    if False:
        while True:
            i = 10
    return foo(s) * 2

def main():
    if False:
        i = 10
        return i + 15
    try:
        bar('0')
    except Exception as e:
        logging.exception(e)
main()
print('END')