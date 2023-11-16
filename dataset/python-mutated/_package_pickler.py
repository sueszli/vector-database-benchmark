"""isort:skip_file"""
from pickle import _compat_pickle, _extension_registry, _getattribute, _Pickler, EXT1, EXT2, EXT4, GLOBAL, Pickler, PicklingError, STACK_GLOBAL
from struct import pack
from types import FunctionType
from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer

class PackagePickler(_Pickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.importer = importer
        super().__init__(*args, **kwargs)
        self.dispatch = _Pickler.dispatch.copy()
        self.dispatch[FunctionType] = PackagePickler.save_global

    def save_global(self, obj, name=None):
        if False:
            print('Hello World!')
        write = self.write
        memo = self.memo
        try:
            (module_name, name) = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from None
        module = self.importer.import_module(module_name)
        (_, parent) = _getattribute(module, name)
        if self.proto >= 2:
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                if code <= 255:
                    write(EXT1 + pack('<B', code))
                elif code <= 65535:
                    write(EXT2 + pack('<H', code))
                else:
                    write(EXT4 + pack('<i', code))
                return
        lastname = name.rpartition('.')[2]
        if parent is module:
            name = lastname
        if self.proto >= 4:
            self.save(module_name)
            self.save(name)
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))
        elif self.proto >= 3:
            write(GLOBAL + bytes(module_name, 'utf-8') + b'\n' + bytes(name, 'utf-8') + b'\n')
        else:
            if self.fix_imports:
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    (module_name, name) = r_name_mapping[module_name, name]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(GLOBAL + bytes(module_name, 'ascii') + b'\n' + bytes(name, 'ascii') + b'\n')
            except UnicodeEncodeError:
                raise PicklingError("can't pickle global identifier '%s.%s' using pickle protocol %i" % (module, name, self.proto)) from None
        self.memoize(obj)

def create_pickler(data_buf, importer, protocol=4):
    if False:
        print('Hello World!')
    if importer is sys_importer:
        return Pickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)