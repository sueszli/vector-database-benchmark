from typing import Any, Iterable
from .version import __version__ as internal_version
__all__ = ['TorchVersion', 'Version', 'InvalidVersion']

class _LazyImport:
    """Wraps around classes lazy imported from packaging.version
    Output of the function v in following snippets are identical:
       from packaging.version import Version
       def v():
           return Version('1.2.3')
    and
       Version = _LazyImport('Version')
       def v():
           return Version('1.2.3')
    The difference here is that in later example imports
    do not happen until v is called
    """

    def __init__(self, cls_name: str) -> None:
        if False:
            i = 10
            return i + 15
        self._cls_name = cls_name

    def get_cls(self):
        if False:
            return 10
        try:
            import packaging.version
        except ImportError:
            from pkg_resources import packaging
        return getattr(packaging.version, self._cls_name)

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.get_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        if False:
            i = 10
            return i + 15
        return isinstance(obj, self.get_cls())
Version = _LazyImport('Version')
InvalidVersion = _LazyImport('InvalidVersion')

class TorchVersion(str):
    """A string with magic powers to compare to both Version and iterables!
    Prior to 1.10.0 torch.__version__ was stored as a str and so many did
    comparisons against torch.__version__ as if it were a str. In order to not
    break them we have TorchVersion which masquerades as a str while also
    having the ability to compare against both packaging.version.Version as
    well as tuples of values, eg. (1, 2, 1)
    Examples:
        Comparing a TorchVersion object to a Version object
            TorchVersion('1.10.0a') > Version('1.10.0a')
        Comparing a TorchVersion object to a Tuple object
            TorchVersion('1.10.0a') > (1, 2)    # 1.2
            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
        Comparing a TorchVersion object against a string
            TorchVersion('1.10.0a') > '1.2'
            TorchVersion('1.10.0a') > '1.2.1'
    """

    def _convert_to_version(self, inp: Any) -> Any:
        if False:
            i = 10
            return i + 15
        if isinstance(inp, Version.get_cls()):
            return inp
        elif isinstance(inp, str):
            return Version(inp)
        elif isinstance(inp, Iterable):
            return Version('.'.join((str(item) for item in inp)))
        else:
            raise InvalidVersion(inp)

    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
        if False:
            return 10
        try:
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            if not isinstance(e, InvalidVersion.get_cls()):
                raise
            return getattr(super(), method)(cmp)
for cmp_method in ['__gt__', '__lt__', '__eq__', '__ge__', '__le__']:
    setattr(TorchVersion, cmp_method, lambda x, y, method=cmp_method: x._cmp_wrapper(y, method))
__version__ = TorchVersion(internal_version)