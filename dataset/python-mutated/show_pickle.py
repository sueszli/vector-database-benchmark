import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO, BinaryIO, Union
__all__ = ['FakeObject', 'FakeClass', 'DumpUnpickler', 'main']

class FakeObject:

    def __init__(self, module, name, args):
        if False:
            i = 10
            return i + 15
        self.module = module
        self.name = name
        self.args = args
        self.state = None

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        state_str = '' if self.state is None else f'(state={self.state!r})'
        return f'{self.module}.{self.name}{self.args!r}{state_str}'

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.state = state

    @staticmethod
    def pp_format(printer, obj, stream, indent, allowance, context, level):
        if False:
            for i in range(10):
                print('nop')
        if not obj.args and obj.state is None:
            stream.write(repr(obj))
            return
        if obj.state is None:
            stream.write(f'{obj.module}.{obj.name}')
            printer._format(obj.args, stream, indent + 1, allowance + 1, context, level)
            return
        if not obj.args:
            stream.write(f'{obj.module}.{obj.name}()(state=\n')
            indent += printer._indent_per_level
            stream.write(' ' * indent)
            printer._format(obj.state, stream, indent, allowance + 1, context, level + 1)
            stream.write(')')
            return
        raise Exception('Need to implement')

class FakeClass:

    def __init__(self, module, name):
        if False:
            print('Hello World!')
        self.module = module
        self.name = name
        self.__new__ = self.fake_new

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.module}.{self.name}'

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return FakeObject(self.module, self.name, args)

    def fake_new(self, *args):
        if False:
            i = 10
            return i + 15
        return FakeObject(self.module, self.name, args[1:])

class DumpUnpickler(pickle._Unpickler):

    def __init__(self, file, *, catch_invalid_utf8=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(file, **kwargs)
        self.catch_invalid_utf8 = catch_invalid_utf8

    def find_class(self, module, name):
        if False:
            print('Hello World!')
        return FakeClass(module, name)

    def persistent_load(self, pid):
        if False:
            for i in range(10):
                print('nop')
        return FakeObject('pers', 'obj', (pid,))
    dispatch = dict(pickle._Unpickler.dispatch)

    def load_binunicode(self):
        if False:
            while True:
                i = 10
        (strlen,) = struct.unpack('<I', self.read(4))
        if strlen > sys.maxsize:
            raise Exception('String too long.')
        str_bytes = self.read(strlen)
        obj: Any
        try:
            obj = str(str_bytes, 'utf-8', 'surrogatepass')
        except UnicodeDecodeError as exn:
            if not self.catch_invalid_utf8:
                raise
            obj = FakeObject('builtin', 'UnicodeDecodeError', (str(exn),))
        self.append(obj)
    dispatch[pickle.BINUNICODE[0]] = load_binunicode

    @classmethod
    def dump(cls, in_stream, out_stream):
        if False:
            print('Hello World!')
        value = cls(in_stream).load()
        pprint.pprint(value, stream=out_stream)
        return value

def main(argv, output_stream=None):
    if False:
        i = 10
        return i + 15
    if len(argv) != 2:
        if output_stream is not None:
            raise Exception('Pass argv of length 2.')
        sys.stderr.write('usage: show_pickle PICKLE_FILE\n')
        sys.stderr.write('  PICKLE_FILE can be any of:\n')
        sys.stderr.write('    path to a pickle file\n')
        sys.stderr.write('    file.zip@member.pkl\n')
        sys.stderr.write('    file.zip@*/pattern.*\n')
        sys.stderr.write('      (shell glob pattern for members)\n')
        sys.stderr.write('      (only first match will be shown)\n')
        return 2
    fname = argv[1]
    handle: Union[IO[bytes], BinaryIO]
    if '@' not in fname:
        with open(fname, 'rb') as handle:
            DumpUnpickler.dump(handle, output_stream)
    else:
        (zfname, mname) = fname.split('@', 1)
        with zipfile.ZipFile(zfname) as zf:
            if '*' not in mname:
                with zf.open(mname) as handle:
                    DumpUnpickler.dump(handle, output_stream)
            else:
                found = False
                for info in zf.infolist():
                    if fnmatch.fnmatch(info.filename, mname):
                        with zf.open(info) as handle:
                            DumpUnpickler.dump(handle, output_stream)
                        found = True
                        break
                if not found:
                    raise Exception(f'Could not find member matching {mname} in {zfname}')
if __name__ == '__main__':
    if True:
        pprint.PrettyPrinter._dispatch[FakeObject.__repr__] = FakeObject.pp_format
    sys.exit(main(sys.argv))