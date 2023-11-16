"""
%store magic for lightweight persistence.

Stores variables, aliases and macros in IPython's database.

To automatically restore stored variables at startup, add this to your
:file:`ipython_config.py` file::

  c.StoreMagics.autorestore = True
"""
import inspect, os, sys, textwrap
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from traitlets import Bool

def restore_aliases(ip, alias=None):
    if False:
        print('Hello World!')
    staliases = ip.db.get('stored_aliases', {})
    if alias is None:
        for (k, v) in staliases.items():
            ip.alias_manager.define_alias(k, v)
    else:
        ip.alias_manager.define_alias(alias, staliases[alias])

def refresh_variables(ip):
    if False:
        return 10
    db = ip.db
    for key in db.keys('autorestore/*'):
        justkey = os.path.basename(key)
        try:
            obj = db[key]
        except KeyError:
            print("Unable to restore variable '%s', ignoring (use %%store -d to forget!)" % justkey)
            print('The error was:', sys.exc_info()[0])
        else:
            ip.user_ns[justkey] = obj

def restore_dhist(ip):
    if False:
        while True:
            i = 10
    ip.user_ns['_dh'] = ip.db.get('dhist', [])

def restore_data(ip):
    if False:
        for i in range(10):
            print('nop')
    refresh_variables(ip)
    restore_aliases(ip)
    restore_dhist(ip)

@magics_class
class StoreMagics(Magics):
    """Lightweight persistence for python variables.

    Provides the %store magic."""
    autorestore = Bool(False, help='If True, any %store-d variables will be automatically restored\n        when IPython starts.\n        ').tag(config=True)

    def __init__(self, shell):
        if False:
            return 10
        super(StoreMagics, self).__init__(shell=shell)
        self.shell.configurables.append(self)
        if self.autorestore:
            restore_data(self.shell)

    @skip_doctest
    @line_magic
    def store(self, parameter_s=''):
        if False:
            for i in range(10):
                print('nop')
        "Lightweight persistence for python variables.\n\n        Example::\n\n          In [1]: l = ['hello',10,'world']\n          In [2]: %store l\n          Stored 'l' (list)\n          In [3]: exit\n\n          (IPython session is closed and started again...)\n\n          ville@badger:~$ ipython\n          In [1]: l\n          NameError: name 'l' is not defined\n          In [2]: %store -r\n          In [3]: l\n          Out[3]: ['hello', 10, 'world']\n\n        Usage:\n\n        * ``%store``          - Show list of all variables and their current\n                                values\n        * ``%store spam bar`` - Store the *current* value of the variables spam\n                                and bar to disk\n        * ``%store -d spam``  - Remove the variable and its value from storage\n        * ``%store -z``       - Remove all variables from storage\n        * ``%store -r``       - Refresh all variables, aliases and directory history\n                                from store (overwrite current vals)\n        * ``%store -r spam bar`` - Refresh specified variables and aliases from store\n                                   (delete current val)\n        * ``%store foo >a.txt``  - Store value of foo to new file a.txt\n        * ``%store foo >>a.txt`` - Append value of foo to file a.txt\n\n        It should be noted that if you change the value of a variable, you\n        need to %store it again if you want to persist the new value.\n\n        Note also that the variables will need to be pickleable; most basic\n        python types can be safely %store'd.\n\n        Also aliases can be %store'd across sessions.\n        To remove an alias from the storage, use the %unalias magic.\n        "
        (opts, argsl) = self.parse_options(parameter_s, 'drz', mode='string')
        args = argsl.split()
        ip = self.shell
        db = ip.db
        if 'd' in opts:
            try:
                todel = args[0]
            except IndexError as e:
                raise UsageError('You must provide the variable to forget') from e
            else:
                try:
                    del db['autorestore/' + todel]
                except BaseException as e:
                    raise UsageError("Can't delete variable '%s'" % todel) from e
        elif 'z' in opts:
            for k in db.keys('autorestore/*'):
                del db[k]
        elif 'r' in opts:
            if args:
                for arg in args:
                    try:
                        obj = db['autorestore/' + arg]
                    except KeyError:
                        try:
                            restore_aliases(ip, alias=arg)
                        except KeyError:
                            print('no stored variable or alias %s' % arg)
                    else:
                        ip.user_ns[arg] = obj
            else:
                restore_data(ip)
        elif not args:
            vars = db.keys('autorestore/*')
            vars.sort()
            if vars:
                size = max(map(len, vars))
            else:
                size = 0
            print('Stored variables and their in-db values:')
            fmt = '%-' + str(size) + 's -> %s'
            get = db.get
            for var in vars:
                justkey = os.path.basename(var)
                print(fmt % (justkey, repr(get(var, '<unavailable>'))[:50]))
        else:
            if len(args) > 1 and args[1].startswith('>'):
                fnam = os.path.expanduser(args[1].lstrip('>').lstrip())
                if args[1].startswith('>>'):
                    fil = open(fnam, 'a', encoding='utf-8')
                else:
                    fil = open(fnam, 'w', encoding='utf-8')
                with fil:
                    obj = ip.ev(args[0])
                    print("Writing '%s' (%s) to file '%s'." % (args[0], obj.__class__.__name__, fnam))
                    if not isinstance(obj, str):
                        from pprint import pprint
                        pprint(obj, fil)
                    else:
                        fil.write(obj)
                        if not obj.endswith('\n'):
                            fil.write('\n')
                return
            for arg in args:
                try:
                    obj = ip.user_ns[arg]
                except KeyError:
                    name = arg
                    try:
                        cmd = ip.alias_manager.retrieve_alias(name)
                    except ValueError as e:
                        raise UsageError("Unknown variable '%s'" % name) from e
                    staliases = db.get('stored_aliases', {})
                    staliases[name] = cmd
                    db['stored_aliases'] = staliases
                    print('Alias stored: %s (%s)' % (name, cmd))
                    return
                else:
                    modname = getattr(inspect.getmodule(obj), '__name__', '')
                    if modname == '__main__':
                        print(textwrap.dedent("                        Warning:%s is %s\n                        Proper storage of interactively declared classes (or instances\n                        of those classes) is not possible! Only instances\n                        of classes in real modules on file system can be %%store'd.\n                        " % (arg, obj)))
                        return
                    db['autorestore/' + arg] = obj
                    print("Stored '%s' (%s)" % (arg, obj.__class__.__name__))

def load_ipython_extension(ip):
    if False:
        i = 10
        return i + 15
    'Load the extension in IPython.'
    ip.register_magics(StoreMagics)