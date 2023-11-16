"""Extract docstrings from Bazaar commands.

This module only handles bzrlib objects that use strings not directly wrapped
by a gettext() call. To generate a complete translation template file, this
output needs to be combined with that of xgettext or a similar command for
extracting those strings, as is done in the bzr Makefile. Sorting the output
is also left to that stage of the process.
"""
from __future__ import absolute_import
import inspect
import os
from bzrlib import commands as _mod_commands, errors, help_topics, option, plugin, help
from bzrlib.trace import mutter, note
from bzrlib.i18n import gettext

def _escape(s):
    if False:
        i = 10
        return i + 15
    s = s.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t').replace('"', '\\"')
    return s

def _normalize(s):
    if False:
        return 10
    lines = s.split('\n')
    if len(lines) == 1:
        s = '"' + _escape(s) + '"'
    else:
        if not lines[-1]:
            del lines[-1]
            lines[-1] = lines[-1] + '\n'
        lines = map(_escape, lines)
        lineterm = '\\n"\n"'
        s = '""\n"' + lineterm.join(lines) + '"'
    return s

def _parse_source(source_text):
    if False:
        i = 10
        return i + 15
    'Get object to lineno mappings from given source_text'
    import ast
    cls_to_lineno = {}
    str_to_lineno = {}
    for node in ast.walk(ast.parse(source_text)):
        if isinstance(node, ast.ClassDef):
            cls_to_lineno[node.name] = node.lineno
        elif isinstance(node, ast.Str):
            str_to_lineno[node.s] = node.lineno - node.s.count('\n')
    return (cls_to_lineno, str_to_lineno)

class _ModuleContext(object):
    """Record of the location within a source tree"""

    def __init__(self, path, lineno=1, _source_info=None):
        if False:
            i = 10
            return i + 15
        self.path = path
        self.lineno = lineno
        if _source_info is not None:
            (self._cls_to_lineno, self._str_to_lineno) = _source_info

    @classmethod
    def from_module(cls, module):
        if False:
            return 10
        'Get new context from module object and parse source for linenos'
        sourcepath = inspect.getsourcefile(module)
        relpath = os.path.relpath(sourcepath)
        return cls(relpath, _source_info=_parse_source(''.join(inspect.findsource(module)[0])))

    def from_class(self, cls):
        if False:
            i = 10
            return i + 15
        'Get new context with same details but lineno of class in source'
        try:
            lineno = self._cls_to_lineno[cls.__name__]
        except (AttributeError, KeyError):
            mutter('Definition of %r not found in %r', cls, self.path)
            return self
        return self.__class__(self.path, lineno, (self._cls_to_lineno, self._str_to_lineno))

    def from_string(self, string):
        if False:
            for i in range(10):
                print('nop')
        'Get new context with same details but lineno of string in source'
        try:
            lineno = self._str_to_lineno[string]
        except (AttributeError, KeyError):
            mutter('String %r not found in %r', string[:20], self.path)
            return self
        return self.__class__(self.path, lineno, (self._cls_to_lineno, self._str_to_lineno))

class _PotExporter(object):
    """Write message details to output stream in .pot file format"""

    def __init__(self, outf, include_duplicates=False):
        if False:
            i = 10
            return i + 15
        self.outf = outf
        if include_duplicates:
            self._msgids = None
        else:
            self._msgids = set()
        self._module_contexts = {}

    def poentry(self, path, lineno, s, comment=None):
        if False:
            for i in range(10):
                print('nop')
        if self._msgids is not None:
            if s in self._msgids:
                return
            self._msgids.add(s)
        if comment is None:
            comment = ''
        else:
            comment = '# %s\n' % comment
        mutter('Exporting msg %r at line %d in %r', s[:20], lineno, path)
        self.outf.write('#: {path}:{lineno}\n{comment}msgid {msg}\nmsgstr ""\n\n'.format(path=path, lineno=lineno, comment=comment, msg=_normalize(s)))

    def poentry_in_context(self, context, string, comment=None):
        if False:
            return 10
        context = context.from_string(string)
        self.poentry(context.path, context.lineno, string, comment)

    def poentry_per_paragraph(self, path, lineno, msgid, include=None):
        if False:
            print('Hello World!')
        paragraphs = msgid.split('\n\n')
        if include is not None:
            paragraphs = filter(include, paragraphs)
        for p in paragraphs:
            self.poentry(path, lineno, p)
            lineno += p.count('\n') + 2

    def get_context(self, obj):
        if False:
            print('Hello World!')
        module = inspect.getmodule(obj)
        try:
            context = self._module_contexts[module.__name__]
        except KeyError:
            context = _ModuleContext.from_module(module)
            self._module_contexts[module.__name__] = context
        if inspect.isclass(obj):
            context = context.from_class(obj)
        return context

def _write_option(exporter, context, opt, note):
    if False:
        for i in range(10):
            print('nop')
    if getattr(opt, 'hidden', False):
        return
    optname = opt.name
    if getattr(opt, 'title', None):
        exporter.poentry_in_context(context, opt.title, 'title of {name!r} {what}'.format(name=optname, what=note))
    for (name, _, _, helptxt) in opt.iter_switches():
        if name != optname:
            if opt.is_hidden(name):
                continue
            name = '='.join([optname, name])
        if helptxt:
            exporter.poentry_in_context(context, helptxt, 'help of {name!r} {what}'.format(name=name, what=note))

def _standard_options(exporter):
    if False:
        while True:
            i = 10
    OPTIONS = option.Option.OPTIONS
    context = exporter.get_context(option)
    for name in sorted(OPTIONS.keys()):
        opt = OPTIONS[name]
        _write_option(exporter, context.from_string(name), opt, 'option')

def _command_options(exporter, context, cmd):
    if False:
        for i in range(10):
            print('nop')
    note = 'option of {0!r} command'.format(cmd.name())
    for opt in cmd.takes_options:
        if not isinstance(opt, str):
            _write_option(exporter, context, opt, note)

def _write_command_help(exporter, cmd):
    if False:
        for i in range(10):
            print('nop')
    context = exporter.get_context(cmd.__class__)
    rawdoc = cmd.__doc__
    dcontext = context.from_string(rawdoc)
    doc = inspect.cleandoc(rawdoc)

    def exclude_usage(p):
        if False:
            for i in range(10):
                print('nop')
        if p.splitlines()[0] != ':Usage:':
            return True
    exporter.poentry_per_paragraph(dcontext.path, dcontext.lineno, doc, exclude_usage)
    _command_options(exporter, context, cmd)

def _command_helps(exporter, plugin_name=None):
    if False:
        print('Hello World!')
    'Extract docstrings from path.\n\n    This respects the Bazaar cmdtable/table convention and will\n    only extract docstrings from functions mentioned in these tables.\n    '
    from glob import glob
    for cmd_name in _mod_commands.builtin_command_names():
        command = _mod_commands.get_cmd_object(cmd_name, False)
        if command.hidden:
            continue
        if plugin_name is not None:
            continue
        note(gettext('Exporting messages from builtin command: %s'), cmd_name)
        _write_command_help(exporter, command)
    plugin_path = plugin.get_core_plugin_path()
    core_plugins = glob(plugin_path + '/*/__init__.py')
    core_plugins = [os.path.basename(os.path.dirname(p)) for p in core_plugins]
    for cmd_name in _mod_commands.plugin_command_names():
        command = _mod_commands.get_cmd_object(cmd_name, False)
        if command.hidden:
            continue
        if plugin_name is not None and command.plugin_name() != plugin_name:
            continue
        if plugin_name is None and command.plugin_name() not in core_plugins:
            continue
        note(gettext('Exporting messages from plugin command: {0} in {1}').format(cmd_name, command.plugin_name()))
        _write_command_help(exporter, command)

def _error_messages(exporter):
    if False:
        print('Hello World!')
    'Extract fmt string from bzrlib.errors.'
    context = exporter.get_context(errors)
    base_klass = errors.BzrError
    for name in dir(errors):
        klass = getattr(errors, name)
        if not inspect.isclass(klass):
            continue
        if not issubclass(klass, base_klass):
            continue
        if klass is base_klass:
            continue
        if klass.internal_error:
            continue
        fmt = getattr(klass, '_fmt', None)
        if fmt:
            note(gettext('Exporting message from error: %s'), name)
            exporter.poentry_in_context(context, fmt)

def _help_topics(exporter):
    if False:
        return 10
    topic_registry = help_topics.topic_registry
    for key in topic_registry.keys():
        doc = topic_registry.get(key)
        if isinstance(doc, str):
            exporter.poentry_per_paragraph('dummy/help_topics/' + key + '/detail.txt', 1, doc)
        elif callable(doc):
            exporter.poentry_per_paragraph('en/help_topics/' + key + '.txt', 1, doc(key))
        summary = topic_registry.get_summary(key)
        if summary is not None:
            exporter.poentry('dummy/help_topics/' + key + '/summary.txt', 1, summary)

def export_pot(outf, plugin=None, include_duplicates=False):
    if False:
        while True:
            i = 10
    exporter = _PotExporter(outf, include_duplicates)
    if plugin is None:
        _standard_options(exporter)
        _command_helps(exporter)
        _error_messages(exporter)
        _help_topics(exporter)
    else:
        _command_helps(exporter, plugin)