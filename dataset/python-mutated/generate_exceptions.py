import argparse
import builtins
import re
import string
import textwrap
from asyncpg.exceptions import _base as apg_exc
_namemap = {'08001': 'ClientCannotConnectError', '08004': 'ConnectionRejectionError', '08006': 'ConnectionFailureError', '38002': 'ModifyingExternalRoutineSQLDataNotPermittedError', '38003': 'ProhibitedExternalRoutineSQLStatementAttemptedError', '38004': 'ReadingExternalRoutineSQLDataNotPermittedError', '39004': 'NullValueInExternalRoutineNotAllowedError', '42000': 'SyntaxOrAccessError', 'XX000': 'InternalServerError'}
_subclassmap = {'0A000': ['InvalidCachedStatementError']}

def _get_error_name(sqlstatename, msgtype, sqlstate):
    if False:
        return 10
    if sqlstate in _namemap:
        return _namemap[sqlstate]
    parts = string.capwords(sqlstatename.replace('_', ' ')).split(' ')
    if parts[-1] in {'Exception', 'Failure'}:
        parts[-1] = 'Error'
    if parts[-1] != 'Error' and msgtype != 'W':
        parts.append('Error')
    for (i, part) in enumerate(parts):
        if part == 'Fdw':
            parts[i] = 'FDW'
        elif part == 'Io':
            parts[i] = 'IO'
        elif part == 'Plpgsql':
            parts[i] = 'PLPGSQL'
        elif part == 'Sql':
            parts[i] = 'SQL'
    errname = ''.join(parts)
    if hasattr(builtins, errname):
        errname = 'Postgres' + errname
    return errname

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='generate _exceptions.py from postgres/errcodes.txt')
    parser.add_argument('errcodesfile', type=str, help='path to errcodes.txt in PostgreSQL source')
    args = parser.parse_args()
    with open(args.errcodesfile, 'r') as errcodes_f:
        errcodes = errcodes_f.read()
    section_re = re.compile('^Section: .*')
    tpl = 'class {clsname}({base}):\n    {docstring}{sqlstate}'
    new_section = True
    section_class = None
    buf = '# GENERATED FROM postgresql/src/backend/utils/errcodes.txt\n' + '# DO NOT MODIFY, use tools/generate_exceptions.py to update\n\n' + 'from ._base import *  # NOQA\nfrom . import _base\n\n\n'
    classes = []
    clsnames = set()

    def _add_class(clsname, base, sqlstate, docstring):
        if False:
            for i in range(10):
                print('nop')
        if sqlstate:
            sqlstate = "sqlstate = '{}'".format(sqlstate)
        else:
            sqlstate = ''
        txt = tpl.format(clsname=clsname, base=base, sqlstate=sqlstate, docstring=docstring)
        if not sqlstate and (not docstring):
            txt += 'pass'
        if len(txt.splitlines()[0]) > 79:
            txt = txt.replace('(', '(\n        ', 1)
        classes.append(txt)
        clsnames.add(clsname)
    for line in errcodes.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        if section_re.match(line):
            new_section = True
            continue
        parts = re.split('\\s+', line)
        if len(parts) < 4:
            continue
        sqlstate = parts[0]
        msgtype = parts[1]
        name = parts[3]
        clsname = _get_error_name(name, msgtype, sqlstate)
        if clsname in {'SuccessfulCompletionError'}:
            continue
        if clsname in clsnames:
            raise ValueError('duplicate exception class name: {}'.format(clsname))
        if new_section:
            section_class = clsname
            if clsname == 'PostgresWarning':
                base = '_base.PostgresLogMessage, Warning'
            elif msgtype == 'W':
                base = 'PostgresWarning'
            else:
                base = '_base.PostgresError'
            new_section = False
        else:
            base = section_class
        existing = apg_exc.PostgresMessageMeta.get_message_class_for_sqlstate(sqlstate)
        if existing and existing is not apg_exc.UnknownPostgresError and existing.__doc__:
            docstring = '"""{}"""\n\n    '.format(existing.__doc__)
        else:
            docstring = ''
        _add_class(clsname=clsname, base=base, sqlstate=sqlstate, docstring=docstring)
        subclasses = _subclassmap.get(sqlstate, [])
        for subclass in subclasses:
            existing = getattr(apg_exc, subclass, None)
            if existing and existing.__doc__:
                docstring = '"""{}"""\n\n    '.format(existing.__doc__)
            else:
                docstring = ''
            _add_class(clsname=subclass, base=clsname, sqlstate=None, docstring=docstring)
    buf += '\n\n\n'.join(classes)
    _all = textwrap.wrap(', '.join(('{!r}'.format(c) for c in sorted(clsnames))))
    buf += '\n\n\n__all__ = (\n    {}\n)'.format('\n    '.join(_all))
    buf += '\n\n__all__ += _base.__all__'
    print(buf)
if __name__ == '__main__':
    main()