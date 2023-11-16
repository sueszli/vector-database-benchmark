"""Dependency scanner for IDL (Interface Definition Language) files."""
from . import ClassicCPP

def IDLScan():
    if False:
        return 10
    'Return a prototype Scanner instance for scanning IDL source files'
    cs = ClassicCPP('IDLScan', '$IDLSUFFIXES', 'CPPPATH', '^[ \\t]*(?:#[ \\t]*include|[ \\t]*import)[ \\t]+(<|")([^>"]+)(>|")')
    return cs