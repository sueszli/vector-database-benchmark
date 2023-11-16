import io, os, sys
if not (hasattr(io, 'IOBase') and hasattr(os, 'mount')):
    print('SKIP')
    raise SystemExit
'\nclass A0:\n    def a0(self): pass\n    def a1(self): pass\n    def a2(self): pass\n    def a3(self): pass\nclass A1:\n    def a0(self): pass\n    def a1(self): pass\n    def a2(self): pass\n    def a3(self): pass\ndef f0():\n    __call__, __class__, __delitem__, __enter__, __exit__, __getattr__, __getitem__,\n    __hash__, __init__, __int__, __iter__, __len__, __main__, __module__, __name__,\n    __new__, __next__, __qualname__, __repr__, __setitem__, __str__,\n    ArithmeticError, AssertionError, AttributeError, BaseException, EOFError, Ellipsis,\n    Exception, GeneratorExit, ImportError, IndentationError, IndexError, KeyError,\n    KeyboardInterrupt, LookupError, MemoryError, NameError, NoneType,\n    NotImplementedError, OSError, OverflowError, RuntimeError, StopIteration,\n    SyntaxError, SystemExit, TypeError, ValueError, ZeroDivisionError,\n    abs, all, any, append, args, bool, builtins, bytearray, bytecode, bytes, callable,\n    chr, classmethod, clear, close, const, copy, count, dict, dir, divmod, end,\n    endswith, eval, exec, extend, find, format, from_bytes, get, getattr, globals,\n    hasattr, hash, id, index, insert, int, isalpha, isdigit, isinstance, islower,\n    isspace, issubclass, isupper, items, iter, join, key, keys, len, list, little,\n    locals, lower, lstrip, main, map, micropython, next, object, open, ord, pop,\n    popitem, pow, print, range, read, readinto, readline, remove, replace, repr,\n    reverse, rfind, rindex, round, rsplit, rstrip, self, send, sep, set, setattr,\n    setdefault, sort, sorted, split, start, startswith, staticmethod, step, stop, str,\n    strip, sum, super, throw, to_bytes, tuple, type, update, upper, value, values,\n    write, zip,\n    name0, name1, name2, name3, name4, name5, name6, name7, name8, name9,\n    quite_a_long_name0, quite_a_long_name1, quite_a_long_name2, quite_a_long_name3,\n    quite_a_long_name4, quite_a_long_name5, quite_a_long_name6, quite_a_long_name7,\n    quite_a_long_name8, quite_a_long_name9, quite_a_long_name10, quite_a_long_name11,\ndef f1():\n    x = "this will be a string object 0"\n    x = "this will be a string object 1"\n    x = "this will be a string object 2"\n    x = "this will be a string object 3"\n    x = "this will be a string object 4"\n    x = "this will be a string object 5"\n    x = "this will be a string object 6"\n    x = "this will be a string object 7"\n    x = "this will be a string object 8"\n    x = "this will be a string object 9"\n    x = b"this will be a bytes object 0"\n    x = b"this will be a bytes object 1"\n    x = b"this will be a bytes object 2"\n    x = b"this will be a bytes object 3"\n    x = b"this will be a bytes object 4"\n    x = b"this will be a bytes object 5"\n    x = b"this will be a bytes object 6"\n    x = b"this will be a bytes object 7"\n    x = b"this will be a bytes object 8"\n    x = b"this will be a bytes object 9"\n    x = ("const tuple 0", None, False, True, 1, 2, 3)\n    x = ("const tuple 1", None, False, True, 1, 2, 3)\n    x = ("const tuple 2", None, False, True, 1, 2, 3)\n    x = ("const tuple 3", None, False, True, 1, 2, 3)\n    x = ("const tuple 4", None, False, True, 1, 2, 3)\n    x = ("const tuple 5", None, False, True, 1, 2, 3)\n    x = ("const tuple 6", None, False, True, 1, 2, 3)\n    x = ("const tuple 7", None, False, True, 1, 2, 3)\n    x = ("const tuple 8", None, False, True, 1, 2, 3)\n    x = ("const tuple 9", None, False, True, 1, 2, 3)\nresult = 123\n'
file_data = b'M\x06\x00\x1f\x81=\x1e\x0etest.py\x00\x0f\x04A0\x00\x04A1\x00\x04f0\x00\x04f1\x00\x0cresult\x00/-5\x04a0\x00\x04a1\x00\x04a2\x00\x04a3\x00\x13\x15\x17\x19\x1b\x1d\x1f!#%\')+1379;=?ACEGIKMOQSUWY[]_acegikmoqsuwy{}\x7f\x81\x01\x81\x03\x81\x05\x81\x07\x81\t\x81\x0b\x81\r\x81\x0f\x81\x11\x81\x13\x81\x15\x81\x17\x81\x19\x81\x1b\x81\x1d\x81\x1f\x81!\x81#\x81%\x81\'\x81)\x81+\x81-\x81/\x811\x813\x815\x817\x819\x81;\x81=\x81?\x81A\x81C\x81E\x81G\x81I\x81K\x81M\x81O\x81Q\x81S\x81U\x81W\x81Y\x81[\x81]\x81_\x81a\x81c\x81e\x81g\x81i\x81k\x81m\x81o\x81q\x81s\x81u\x81w\x81y\x81{\x81}\x81\x7f\x82\x01\x82\x03\x82\x05\x82\x07\x82\t\x82\x0b\x82\r\x82\x0f\x82\x11\x82\x13\x82\x15\x82\x17\x82\x19\x82\x1b\x82\x1d\x82\x1f\x82!\x82#\x82%\x82\'\x82)\x82+\x82-\x82/\x821\x823\x825\x827\x829\x82;\x82=\x82?\x82A\x82E\x82G\x82I\x82K\nname0\x00\nname1\x00\nname2\x00\nname3\x00\nname4\x00\nname5\x00\nname6\x00\nname7\x00\nname8\x00\nname9\x00$quite_a_long_name0\x00$quite_a_long_name1\x00$quite_a_long_name2\x00$quite_a_long_name3\x00$quite_a_long_name4\x00$quite_a_long_name5\x00$quite_a_long_name6\x00$quite_a_long_name7\x00$quite_a_long_name8\x00$quite_a_long_name9\x00&quite_a_long_name10\x00&quite_a_long_name11\x00\x05\x1ethis will be a string object 0\x00\x05\x1ethis will be a string object 1\x00\x05\x1ethis will be a string object 2\x00\x05\x1ethis will be a string object 3\x00\x05\x1ethis will be a string object 4\x00\x05\x1ethis will be a string object 5\x00\x05\x1ethis will be a string object 6\x00\x05\x1ethis will be a string object 7\x00\x05\x1ethis will be a string object 8\x00\x05\x1ethis will be a string object 9\x00\x06\x1dthis will be a bytes object 0\x00\x06\x1dthis will be a bytes object 1\x00\x06\x1dthis will be a bytes object 2\x00\x06\x1dthis will be a bytes object 3\x00\x06\x1dthis will be a bytes object 4\x00\x06\x1dthis will be a bytes object 5\x00\x06\x1dthis will be a bytes object 6\x00\x06\x1dthis will be a bytes object 7\x00\x06\x1dthis will be a bytes object 8\x00\x06\x1dthis will be a bytes object 9\x00\n\x07\x05\rconst tuple 0\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 1\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 2\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 3\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 4\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 5\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 6\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 7\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 8\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\n\x07\x05\rconst tuple 9\x00\x01\x02\x03\x07\x011\x07\x012\x07\x013\x82d\x10\x12\x01i@i@\x84\x18\x84\x1fT2\x00\x10\x024\x02\x16\x02T2\x01\x10\x034\x02\x16\x032\x02\x16\x042\x03\x16\x05"\x80{\x16\x06Qc\x04\x82\x0c\x00\n\x02($$$\x11\x07\x16\x08\x10\x02\x16\t2\x00\x16\n2\x01\x16\x0b2\x02\x16\x0c2\x03\x16\rQc\x04@\t\x08\n\x81\x0b Qc@\t\x08\x0b\x81\x0b@Qc@\t\x08\x0c\x81\x0b`QcH\t\n\r\x81\x0b` Qc\x82\x14\x00\x0c\x03h`$$$\x11\x07\x16\x08\x10\x03\x16\t2\x00\x16\n2\x01\x16\x0b2\x02\x16\x0c2\x03\x16\rQc\x04H\t\n\n\x81\x0b``QcH\t\n\x0b\x81\x0b\x80\x07QcH\t\n\x0c\x81\x0b\x80\x08QcH\t\n\r\x81\x0b\x80\tQc\xa08P:\x04\x80\x0b13///---997799<\x1f%\x1f"\x1f%)\x1f"//\x12\x0e\x12\x0f\x12\x10\x12\x11\x12\x12\x12\x13\x12\x14*\x07Y\x12\x15\x12\x16\x12\x17\x12\x18\x12\x19\x12\x1a\x12\x08\x12\x07*\x08Y\x12\x1b\x12\x1c\x12\t\x12\x1d\x12\x1e\x12\x1f*\x06Y\x12 \x12!\x12"\x12#\x12$\x12%*\x06Y\x12&\x12\'\x12(\x12)\x12*\x12+*\x06Y\x12,\x12-\x12.\x12/\x120*\x05Y\x121\x122\x123\x124\x125*\x05Y\x126\x127\x128\x129\x12:*\x05Y\x12;\x12<\x12=\x12>\x12?\x12@\x12A\x12B\x12C\x12D\x12E*\x0bY\x12F\x12G\x12H\x12I\x12J\x12K\x12L\x12M\x12N\x12O\x12P*\x0bY\x12Q\x12R\x12S\x12T\x12U\x12V\x12W\x12X\x12Y\x12Z*\nY\x12[\x12\\\x12]\x12^\x12_\x12`\x12a\x12b\x12c\x12d*\nY\x12e\x12f\x12g\x12h\x12i\x12j\x12k\x12l\x12m\x12n\x12o*\x0bY\x12p\x12q\x12r\x12s\x12t\x12u\x12v\x12w\x12x\x12y\x12z*\x0bY\x12{\x12|\x12}\x12~\x12\x7f\x12\x81\x00\x12\x81\x01\x12\x81\x02\x12\x81\x03\x12\x81\x04*\nY\x12\x81\x05\x12\x81\x06\x12\x81\x07\x12\x81\x08\x12\x81\t\x12\x81\n\x12\x81\x0b\x12\x81\x0c\x12\x81\r\x12\x81\x0e\x12\x81\x0f*\x0bY\x12\x81\x10\x12\x81\x11\x12\x81\x12\x12\x81\x13\x12\x81\x14\x12\x81\x15\x12\x81\x16\x12\x81\x17\x12\x81\x18\x12\x81\x19*\nY\x12\x81\x1a\x12\x81\x1b\x12\x81\x1c\x12\x81\x1d\x12\x81\x1e\x12\x81\x1f\x12\x81 \x12\x81!\x12\x81"\x12\x81#\x12\x81$*\x0bY\x12\x81%\x12\x81&*\x02Y\x12\x81\'\x12\x81(\x12\x81)\x12\x81*\x12\x81+\x12\x81,\x12\x81-\x12\x81.\x12\x81/\x12\x810*\nY\x12\x811\x12\x812\x12\x813\x12\x814*\x04Y\x12\x815\x12\x816\x12\x817\x12\x818*\x04Y\x12\x819\x12\x81:\x12\x81;\x12\x81<*\x04YQc\x87p\x08@\x05\x80###############################\x00\xc0#\x01\xc0#\x02\xc0#\x03\xc0#\x04\xc0#\x05\xc0#\x06\xc0#\x07\xc0#\x08\xc0#\t\xc0#\n\xc0#\x0b\xc0#\x0c\xc0#\r\xc0#\x0e\xc0#\x0f\xc0#\x10\xc0#\x11\xc0#\x12\xc0#\x13\xc0#\x14\xc0#\x15\xc0#\x16\xc0#\x17\xc0#\x18\xc0#\x19\xc0#\x1a\xc0#\x1b\xc0#\x1c\xc0#\x1d\xc0Qc'

class File(io.IOBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.off = 0

    def ioctl(self, request, arg):
        if False:
            return 10
        return 0

    def readinto(self, buf):
        if False:
            for i in range(10):
                print('nop')
        buf[:] = memoryview(file_data)[self.off:self.off + len(buf)]
        self.off += len(buf)
        return len(buf)

class FS:

    def mount(self, readonly, mkfs):
        if False:
            print('Hello World!')
        pass

    def chdir(self, path):
        if False:
            return 10
        pass

    def stat(self, path):
        if False:
            return 10
        if path == '/__injected.mpy':
            return tuple((0 for _ in range(10)))
        else:
            raise OSError(-2)

    def open(self, path, mode):
        if False:
            return 10
        return File()

def mount():
    if False:
        i = 10
        return i + 15
    os.mount(FS(), '/__remote')
    sys.path.insert(0, '/__remote')

def test():
    if False:
        for i in range(10):
            print('nop')
    global result
    module = __import__('__injected')
    result = module.result
bm_params = {(1, 1): ()}

def bm_setup(params):
    if False:
        while True:
            i = 10
    mount()
    return (lambda : test(), lambda : (1, result))