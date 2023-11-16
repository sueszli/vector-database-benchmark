"""Functions for handling trustdb and trust calculations.

The functions within this module take an instance of :class:`gnupg.GPGBase` or
a suitable subclass as their first argument.
"""
import os
from . import _util
from ._util import log

def _create_trustdb(cls):
    if False:
        i = 10
        return i + 15
    "Create the trustdb file in our homedir, if it doesn't exist."
    trustdb = os.path.join(cls.homedir, 'trustdb.gpg')
    if not os.path.isfile(trustdb):
        log.info('GnuPG complained that your trustdb file was missing. {}'.format('This is likely due to changing to a new homedir.'))
        log.info('Creating trustdb.gpg file in your GnuPG homedir.')
        cls.fix_trustdb(trustdb)

def export_ownertrust(cls, trustdb=None):
    if False:
        for i in range(10):
            print('nop')
    "Export ownertrust to a trustdb file.\n\n    If there is already a file named :file:`trustdb.gpg` in the current GnuPG\n    homedir, it will be renamed to :file:`trustdb.gpg.bak`.\n\n    :param string trustdb: The path to the trustdb.gpg file. If not given,\n                           defaults to ``'trustdb.gpg'`` in the current GnuPG\n                           homedir.\n    "
    if trustdb is None:
        trustdb = os.path.join(cls.homedir, 'trustdb.gpg')
    try:
        os.rename(trustdb, trustdb + '.bak')
    except OSError as err:
        log.debug(str(err))
    export_proc = cls._open_subprocess(['--export-ownertrust'])
    tdb = open(trustdb, 'wb')
    _util._threaded_copy_data(export_proc.stdout, tdb)
    export_proc.wait()

def import_ownertrust(cls, trustdb=None):
    if False:
        i = 10
        return i + 15
    'Import ownertrust from a trustdb file.\n\n    :param str trustdb: The path to the trustdb.gpg file. If not given,\n                        defaults to :file:`trustdb.gpg` in the current GnuPG\n                        homedir.\n    '
    if trustdb is None:
        trustdb = os.path.join(cls.homedir, 'trustdb.gpg')
    import_proc = cls._open_subprocess(['--import-ownertrust'])
    try:
        tdb = open(trustdb, 'rb')
    except OSError:
        log.error('trustdb file %s does not exist!' % trustdb)
    _util._threaded_copy_data(tdb, import_proc.stdin)
    import_proc.wait()

def fix_trustdb(cls, trustdb=None):
    if False:
        for i in range(10):
            print('nop')
    "Attempt to repair a broken trustdb.gpg file.\n\n    GnuPG>=2.0.x has this magical-seeming flag: `--fix-trustdb`. You'd think\n    it would fix the the trustdb. Hah! It doesn't. Here's what it does\n    instead::\n\n      (gpg)~/code/python-gnupg $ gpg2 --fix-trustdb\n      gpg: You may try to re-create the trustdb using the commands:\n      gpg:   cd ~/.gnupg\n      gpg:   gpg2 --export-ownertrust > otrust.tmp\n      gpg:   rm trustdb.gpg\n      gpg:   gpg2 --import-ownertrust < otrust.tmp\n      gpg: If that does not work, please consult the manual\n\n    Brilliant piece of software engineering right there.\n\n    :param str trustdb: The path to the trustdb.gpg file. If not given,\n                        defaults to :file:`trustdb.gpg` in the current GnuPG\n                        homedir.\n    "
    if trustdb is None:
        trustdb = os.path.join(cls.homedir, 'trustdb.gpg')
    export_proc = cls._open_subprocess(['--export-ownertrust'])
    import_proc = cls._open_subprocess(['--import-ownertrust'])
    _util._threaded_copy_data(export_proc.stdout, import_proc.stdin)
    export_proc.wait()
    import_proc.wait()