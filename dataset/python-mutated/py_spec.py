"""A Python specification is an abstract requirement definition of an interpreter."""
from __future__ import annotations
import contextlib
import os
import re
from collections import OrderedDict
from virtualenv.info import fs_is_case_sensitive
PATTERN = re.compile('^(?P<impl>[a-zA-Z]+)?(?P<version>[0-9.]+)?(?:-(?P<arch>32|64))?$')

class PythonSpec:
    """Contains specification about a Python Interpreter."""

    def __init__(self, str_spec, implementation, major, minor, micro, architecture, path) -> None:
        if False:
            return 10
        self.str_spec = str_spec
        self.implementation = implementation
        self.major = major
        self.minor = minor
        self.micro = micro
        self.architecture = architecture
        self.path = path

    @classmethod
    def from_string_spec(cls, string_spec):
        if False:
            return 10
        (impl, major, minor, micro, arch, path) = (None, None, None, None, None, None)
        if os.path.isabs(string_spec):
            path = string_spec
        else:
            ok = False
            match = re.match(PATTERN, string_spec)
            if match:

                def _int_or_none(val):
                    if False:
                        i = 10
                        return i + 15
                    return None if val is None else int(val)
                try:
                    groups = match.groupdict()
                    version = groups['version']
                    if version is not None:
                        versions = tuple((int(i) for i in version.split('.') if i))
                        if len(versions) > 3:
                            raise ValueError
                        if len(versions) == 3:
                            (major, minor, micro) = versions
                        elif len(versions) == 2:
                            (major, minor) = versions
                        elif len(versions) == 1:
                            version_data = versions[0]
                            major = int(str(version_data)[0])
                            if version_data > 9:
                                minor = int(str(version_data)[1:])
                    ok = True
                except ValueError:
                    pass
                else:
                    impl = groups['impl']
                    if impl in {'py', 'python'}:
                        impl = None
                    arch = _int_or_none(groups['arch'])
            if not ok:
                path = string_spec
        return cls(string_spec, impl, major, minor, micro, arch, path)

    def generate_names(self):
        if False:
            i = 10
            return i + 15
        impls = OrderedDict()
        if self.implementation:
            impls[self.implementation] = False
            if fs_is_case_sensitive():
                impls[self.implementation.lower()] = False
                impls[self.implementation.upper()] = False
        impls['python'] = True
        version = (self.major, self.minor, self.micro)
        with contextlib.suppress(ValueError):
            version = version[:version.index(None)]
        for (impl, match) in impls.items():
            for at in range(len(version), -1, -1):
                cur_ver = version[0:at]
                spec = f"{impl}{'.'.join((str(i) for i in cur_ver))}"
                yield (spec, match)

    @property
    def is_abs(self):
        if False:
            print('Hello World!')
        return self.path is not None and os.path.isabs(self.path)

    def satisfies(self, spec):
        if False:
            return 10
        "Called when there's a candidate metadata spec to see if compatible - e.g. PEP-514 on Windows."
        if spec.is_abs and self.is_abs and (self.path != spec.path):
            return False
        if spec.implementation is not None and spec.implementation.lower() != self.implementation.lower():
            return False
        if spec.architecture is not None and spec.architecture != self.architecture:
            return False
        for (our, req) in zip((self.major, self.minor, self.micro), (spec.major, spec.minor, spec.micro)):
            if req is not None and our is not None and (our != req):
                return False
        return True

    def __repr__(self) -> str:
        if False:
            return 10
        name = type(self).__name__
        params = ('implementation', 'major', 'minor', 'micro', 'architecture', 'path')
        return f"{name}({', '.join((f'{k}={getattr(self, k)}' for k in params if getattr(self, k) is not None))})"
__all__ = ['PythonSpec']