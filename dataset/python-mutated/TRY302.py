"""
Violation:

Checks for uses of `raise` directly after a `rescue`.
"""

class MyException(Exception):
    pass

def bad():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except Exception:
        raise

def bad():
    if False:
        while True:
            i = 10
    try:
        process()
    except Exception:
        raise
        print('this code is pointless!')

def bad():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except:
        raise

def bad():
    if False:
        print('Hello World!')
    try:
        process()
    except Exception:
        raise

def bad():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except Exception as e:
        raise

def bad():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except Exception as e:
        raise e

def bad():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException:
        raise
    except Exception:
        raise

def bad():
    if False:
        while True:
            i = 10
    try:
        process()
    except MyException as e:
        raise e
    except Exception as e:
        raise e

def bad():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException as ex:
        raise ex
    except Exception as e:
        raise e

def fine():
    if False:
        while True:
            i = 10
    try:
        process()
    except Exception as e:
        raise e from None

def fine():
    if False:
        while True:
            i = 10
    try:
        process()
    except Exception as e:
        raise e from Exception

def fine():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except Exception as e:
        raise ex

def fine():
    if False:
        print('Hello World!')
    try:
        process()
    except MyException:
        raise
    except Exception:
        print('bar')

def fine():
    if False:
        return 10
    try:
        process()
    except Exception:
        print('initiating rapid unscheduled disassembly of program')

def fine():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException:
        print('oh no!')
        raise

def fine():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except Exception:
        if True:
            raise

def fine():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    finally:
        print('but i am a statement')
        raise