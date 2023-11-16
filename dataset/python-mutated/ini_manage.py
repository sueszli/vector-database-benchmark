"""
Edit ini files

:maintainer: <akilesh1597@gmail.com>
:maturity: new
:depends: re
:platform: all

(for example /etc/sysctl.conf)
"""
import logging
import os
import re
import salt.utils.data
import salt.utils.files
import salt.utils.json
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
from salt.utils.odict import OrderedDict
log = logging.getLogger(__name__)
__virtualname__ = 'ini'

def __virtual__():
    if False:
        return 10
    '\n    Rename to ini\n    '
    return __virtualname__
INI_REGX = re.compile('^\\s*\\[(.+?)\\]\\s*$', flags=re.M)
COM_REGX = re.compile('^\\s*(#|;)\\s*(.*)')
INDENTED_REGX = re.compile('(\\s+)(.*)')

def set_option(file_name, sections=None, separator='='):
    if False:
        i = 10
        return i + 15
    '\n    Edit an ini file, replacing one or more sections. Returns a dictionary\n    containing the changes made.\n\n    file_name\n        path of ini_file\n\n    sections : None\n        A dictionary representing the sections to be edited ini file\n        The keys are the section names and the values are the dictionary\n        containing the options\n        If the ini file does not contain sections the keys and values represent\n        the options\n\n    separator : =\n        A character used to separate keys and values. Standard ini files use\n        the "=" character.\n\n        .. versionadded:: 2016.11.0\n\n    API Example:\n\n    .. code-block:: python\n\n        import salt.client\n        with salt.client.get_local_client() as sc:\n            sc.cmd(\n                \'target\', \'ini.set_option\',\n                [\'path_to_ini_file\', \'{"section_to_change": {"key": "value"}}\']\n            )\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ini.set_option /path/to/ini \'{section_foo: {key: value}}\'\n    '
    sections = sections or {}
    changes = {}
    inifile = _Ini.get_ini_file(file_name, separator=separator)
    changes = inifile.update(sections)
    inifile.flush()
    return changes

def get_option(file_name, section, option, separator='='):
    if False:
        i = 10
        return i + 15
    "\n    Get value of a key from a section in an ini file. Returns ``None`` if\n    no matching key was found.\n\n    API Example:\n\n    .. code-block:: python\n\n        import salt.client\n        with salt.client.get_local_client() as sc:\n            sc.cmd('target', 'ini.get_option',\n                   [path_to_ini_file, section_name, option])\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ini.get_option /path/to/ini section_name option_name\n    "
    inifile = _Ini.get_ini_file(file_name, separator=separator)
    if section:
        try:
            return inifile.get(section, {}).get(option, None)
        except AttributeError:
            return None
    else:
        return inifile.get(option, None)

def remove_option(file_name, section, option, separator='='):
    if False:
        print('Hello World!')
    "\n    Remove a key/value pair from a section in an ini file. Returns the value of\n    the removed key, or ``None`` if nothing was removed.\n\n    API Example:\n\n    .. code-block:: python\n\n        import salt\n        sc = salt.client.get_local_client()\n        sc.cmd('target', 'ini.remove_option',\n               [path_to_ini_file, section_name, option])\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ini.remove_option /path/to/ini section_name option_name\n    "
    inifile = _Ini.get_ini_file(file_name, separator=separator)
    if isinstance(inifile.get(section), (dict, OrderedDict)):
        value = inifile.get(section, {}).pop(option, None)
    else:
        value = inifile.pop(option, None)
    inifile.flush()
    return value

def get_section(file_name, section, separator='='):
    if False:
        return 10
    "\n    Retrieve a section from an ini file. Returns the section as dictionary. If\n    the section is not found, an empty dictionary is returned.\n\n    API Example:\n\n    .. code-block:: python\n\n        import salt.client\n        with salt.client.get_local_client() as sc:\n            sc.cmd('target', 'ini.get_section',\n                   [path_to_ini_file, section_name])\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ini.get_section /path/to/ini section_name\n    "
    inifile = _Ini.get_ini_file(file_name, separator=separator)
    ret = {}
    for (key, value) in inifile.get(section, {}).items():
        if key[0] != '#':
            ret.update({key: value})
    return ret

def remove_section(file_name, section, separator='='):
    if False:
        while True:
            i = 10
    "\n    Remove a section in an ini file. Returns the removed section as dictionary,\n    or ``None`` if nothing was removed.\n\n    API Example:\n\n    .. code-block:: python\n\n        import salt.client\n        with  salt.client.get_local_client() as sc:\n            sc.cmd('target', 'ini.remove_section',\n                   [path_to_ini_file, section_name])\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ini.remove_section /path/to/ini section_name\n    "
    inifile = _Ini.get_ini_file(file_name, separator=separator)
    if section in inifile:
        section = inifile.pop(section)
        inifile.flush()
        ret = {}
        for (key, value) in section.items():
            if key[0] != '#':
                ret.update({key: value})
        return ret

def get_ini(file_name, separator='='):
    if False:
        i = 10
        return i + 15
    "\n    Retrieve whole structure from an ini file and return it as dictionary.\n\n    API Example:\n\n    .. code-block:: python\n\n        import salt.client\n        with salt.client.giet_local_client() as sc:\n            sc.cmd('target', 'ini.get_ini',\n                   [path_to_ini_file])\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ini.get_ini /path/to/ini\n    "

    def ini_odict2dict(odict):
        if False:
            i = 10
            return i + 15
        '\n        Transform OrderedDict to regular dict recursively\n        :param odict: OrderedDict\n        :return: regular dict\n        '
        ret = {}
        for (key, val) in odict.items():
            if key[0] != '#':
                if isinstance(val, (dict, OrderedDict)):
                    ret.update({key: ini_odict2dict(val)})
                else:
                    ret.update({key: val})
        return ret
    inifile = _Ini.get_ini_file(file_name, separator=separator)
    return ini_odict2dict(inifile)

class _Section(OrderedDict):

    def __init__(self, name, inicontents='', separator='=', commenter='#'):
        if False:
            i = 10
            return i + 15
        super().__init__(self)
        self.name = name
        self.inicontents = inicontents
        self.sep = separator
        self.com = commenter
        opt_regx_prefix = '(\\s*)(.+?)\\s*'
        opt_regx_suffix = '\\s*(.*)\\s*'
        self.opt_regx_str = '{}(\\{}){}'.format(opt_regx_prefix, self.sep, opt_regx_suffix)
        self.opt_regx = re.compile(self.opt_regx_str)

    def refresh(self, inicontents=None):
        if False:
            while True:
                i = 10
        comment_count = 1
        unknown_count = 1
        curr_indent = ''
        inicontents = inicontents or self.inicontents
        inicontents = inicontents.strip(os.linesep)
        if not inicontents:
            return
        for opt in self:
            self.pop(opt)
        for opt_str in inicontents.split(os.linesep):
            com_match = COM_REGX.match(opt_str)
            if com_match:
                name = '#comment{}'.format(comment_count)
                self.com = com_match.group(1)
                comment_count += 1
                self.update({name: opt_str})
                continue
            indented_match = INDENTED_REGX.match(opt_str)
            if indented_match:
                indent = indented_match.group(1).replace('\t', '    ')
                if indent > curr_indent:
                    options = list(self)
                    if options:
                        prev_opt = options[-1]
                        value = self.get(prev_opt)
                        self.update({prev_opt: os.linesep.join((value, opt_str))})
                    continue
            opt_match = self.opt_regx.match(opt_str)
            if opt_match:
                (curr_indent, name, self.sep, value) = opt_match.groups()
                curr_indent = curr_indent.replace('\t', '    ')
                self.update({name: value})
                continue
            name = '#unknown{}'.format(unknown_count)
            self.update({name: opt_str})
            unknown_count += 1

    def _uncomment_if_commented(self, opt_key):
        if False:
            for i in range(10):
                print('nop')
        options_backup = OrderedDict()
        comment_index = None
        for (key, value) in self.items():
            if comment_index is not None:
                options_backup.update({key: value})
                continue
            if '#comment' not in key:
                continue
            opt_match = self.opt_regx.match(value.lstrip('#'))
            if opt_match and opt_match.group(2) == opt_key:
                comment_index = key
        for key in options_backup:
            self.pop(key)
        self.pop(comment_index, None)
        super().update({opt_key: None})
        for (key, value) in options_backup.items():
            super().update({key: value})

    def update(self, update_dict):
        if False:
            print('Hello World!')
        changes = {}
        for (key, value) in update_dict.items():
            if isinstance(value, (dict, OrderedDict)):
                sect = _Section(name=key, inicontents='', separator=self.sep, commenter=self.com)
                sect.update(value)
                value = sect
                value_plain = value.as_dict()
            else:
                value = str(value)
                value_plain = value
            if key not in self:
                changes.update({key: {'before': None, 'after': value_plain}})
                if not isinstance(value, _Section):
                    self._uncomment_if_commented(key)
                super().update({key: value})
            else:
                curr_value = self.get(key, None)
                if isinstance(curr_value, _Section):
                    sub_changes = curr_value.update(value)
                    if sub_changes:
                        changes.update({key: sub_changes})
                elif curr_value != value:
                    changes.update({key: {'before': curr_value, 'after': value_plain}})
                    super().update({key: value})
        return changes

    def gen_ini(self):
        if False:
            i = 10
            return i + 15
        yield '{0}[{1}]{0}'.format(os.linesep, self.name)
        sections_dict = OrderedDict()
        for (name, value) in self.items():
            if COM_REGX.match(name):
                yield '{}{}'.format(value, os.linesep)
            elif isinstance(value, _Section):
                sections_dict.update({name: value})
            else:
                yield '{}{}{}{}'.format(name, ' {} '.format(self.sep) if self.sep != ' ' else self.sep, value, os.linesep)
        for (name, value) in sections_dict.items():
            yield from value.gen_ini()

    def as_ini(self):
        if False:
            i = 10
            return i + 15
        return ''.join(self.gen_ini())

    def as_dict(self):
        if False:
            print('Hello World!')
        return dict(self)

    def dump(self):
        if False:
            i = 10
            return i + 15
        print(str(self))

    def __repr__(self, _repr_running=None):
        if False:
            return 10
        _repr_running = _repr_running or {}
        try:
            super_repr = super().__repr__(_repr_running)
        except TypeError:
            super_repr = super().__repr__()
        return os.linesep.join((super_repr, salt.utils.json.dumps(self, indent=4)))

    def __str__(self):
        if False:
            print('Hello World!')
        return salt.utils.json.dumps(self, indent=4)

    def __eq__(self, item):
        if False:
            print('Hello World!')
        return isinstance(item, self.__class__) and self.name == item.name

    def __ne__(self, item):
        if False:
            print('Hello World!')
        return not (isinstance(item, self.__class__) and self.name == item.name)

class _Ini(_Section):

    def refresh(self, inicontents=None):
        if False:
            i = 10
            return i + 15
        if inicontents is None:
            if not os.path.exists(self.name):
                log.trace('File %s does not exist and will be created', self.name)
                return
            try:
                with salt.utils.files.fopen(self.name) as rfh:
                    inicontents = salt.utils.stringutils.to_unicode(rfh.read())
                    inicontents = os.linesep.join(inicontents.splitlines())
            except OSError as exc:
                if __opts__['test'] is False:
                    raise CommandExecutionError("Unable to open file '{}'. Exception: {}".format(self.name, exc))
        if not inicontents:
            return
        self.clear()
        inicontents = INI_REGX.split(inicontents)
        inicontents.reverse()
        super().refresh(inicontents.pop())
        for (section_name, sect_ini) in self._gen_tuples(inicontents):
            try:
                sect_obj = _Section(section_name, sect_ini, separator=self.sep)
                sect_obj.refresh()
                self.update({sect_obj.name: sect_obj})
            except StopIteration:
                pass

    def flush(self):
        if False:
            i = 10
            return i + 15
        try:
            with salt.utils.files.fopen(self.name, 'wb') as outfile:
                ini_gen = self.gen_ini()
                next(ini_gen)
                ini_gen_list = list(ini_gen)
                if ini_gen_list:
                    ini_gen_list[0] = ini_gen_list[0].lstrip(os.linesep)
                outfile.writelines(salt.utils.data.encode(ini_gen_list))
        except OSError as exc:
            raise CommandExecutionError("Unable to write file '{}'. Exception: {}".format(self.name, exc))

    @staticmethod
    def get_ini_file(file_name, separator='='):
        if False:
            print('Hello World!')
        inifile = _Ini(file_name, separator=separator)
        inifile.refresh()
        return inifile

    @staticmethod
    def _gen_tuples(list_object):
        if False:
            while True:
                i = 10
        while True:
            try:
                key = list_object.pop()
                value = list_object.pop()
            except IndexError:
                return
            else:
                yield (key, value)