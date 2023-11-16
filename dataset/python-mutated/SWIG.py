"""Dependency scanner for SWIG code."""
from . import ClassicCPP
SWIGSuffixes = ['.i']

def SWIGScanner():
    if False:
        while True:
            i = 10
    expr = '^[ \\t]*%[ \\t]*(?:include|import|extern)[ \\t]*(<|"?)([^>\\s"]+)(?:>|"?)'
    scanner = ClassicCPP('SWIGScanner', '.i', 'SWIGPATH', expr)
    return scanner