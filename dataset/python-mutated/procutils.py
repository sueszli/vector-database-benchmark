"""
Utilities for dealing with processes.
"""
import os

def which(name, flags=os.X_OK):
    if False:
        return 10
    '\n    Search PATH for executable files with the given name.\n\n    On newer versions of MS-Windows, the PATHEXT environment variable will be\n    set to the list of file extensions for files considered executable. This\n    will normally include things like ".EXE". This function will also find files\n    with the given name ending with any of these extensions.\n\n    On MS-Windows the only flag that has any meaning is os.F_OK. Any other\n    flags will be ignored.\n\n    @type name: C{str}\n    @param name: The name for which to search.\n\n    @type flags: C{int}\n    @param flags: Arguments to L{os.access}.\n\n    @rtype: C{list}\n    @return: A list of the full paths to files found, in the order in which they\n    were found.\n    '
    result = []
    exts = list(filter(None, os.environ.get('PATHEXT', '').split(os.pathsep)))
    path = os.environ.get('PATH', None)
    if path is None:
        return []
    for p in os.environ.get('PATH', '').split(os.pathsep):
        p = os.path.join(p, name)
        if os.access(p, flags):
            result.append(p)
        for e in exts:
            pext = p + e
            if os.access(pext, flags):
                result.append(pext)
    return result