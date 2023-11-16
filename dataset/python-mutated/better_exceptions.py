from loguru import logger

def division(a, b):
    if False:
        print('Hello World!')
    return a / b

def nested(c):
    if False:
        while True:
            i = 10
    try:
        division(1, c)
    except ZeroDivisionError:
        logger.exception('ZeroDivisionError')
if __name__ == '__main__':
    nested(0)