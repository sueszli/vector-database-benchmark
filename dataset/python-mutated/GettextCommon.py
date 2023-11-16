"""SCons.Tool.GettextCommon module

Used by several tools of `gettext` toolset.
"""
__revision__ = 'src/engine/SCons/Tool/GettextCommon.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Warnings
import re

class XgettextToolWarning(SCons.Warnings.Warning):
    pass

class XgettextNotFound(XgettextToolWarning):
    pass

class MsginitToolWarning(SCons.Warnings.Warning):
    pass

class MsginitNotFound(MsginitToolWarning):
    pass

class MsgmergeToolWarning(SCons.Warnings.Warning):
    pass

class MsgmergeNotFound(MsgmergeToolWarning):
    pass

class MsgfmtToolWarning(SCons.Warnings.Warning):
    pass

class MsgfmtNotFound(MsgfmtToolWarning):
    pass
SCons.Warnings.enableWarningClass(XgettextToolWarning)
SCons.Warnings.enableWarningClass(XgettextNotFound)
SCons.Warnings.enableWarningClass(MsginitToolWarning)
SCons.Warnings.enableWarningClass(MsginitNotFound)
SCons.Warnings.enableWarningClass(MsgmergeToolWarning)
SCons.Warnings.enableWarningClass(MsgmergeNotFound)
SCons.Warnings.enableWarningClass(MsgfmtToolWarning)
SCons.Warnings.enableWarningClass(MsgfmtNotFound)

class _POTargetFactory(object):
    """ A factory of `PO` target files.

    Factory defaults differ from these of `SCons.Node.FS.FS`.  We set `precious`
    (this is required by builders and actions gettext) and `noclean` flags by
    default for all produced nodes.
    """

    def __init__(self, env, nodefault=True, alias=None, precious=True, noclean=True):
        if False:
            while True:
                i = 10
        " Object constructor.\n\n        **Arguments**\n\n            - *env* (`SCons.Environment.Environment`)\n            - *nodefault* (`boolean`) - if `True`, produced nodes will be ignored\n              from default target `'.'`\n            - *alias* (`string`) - if provided, produced nodes will be automatically\n              added to this alias, and alias will be set as `AlwaysBuild`\n            - *precious* (`boolean`) - if `True`, the produced nodes will be set as\n              `Precious`.\n            - *noclen* (`boolean`) - if `True`, the produced nodes will be excluded\n              from `Clean`.\n        "
        self.env = env
        self.alias = alias
        self.precious = precious
        self.noclean = noclean
        self.nodefault = nodefault

    def _create_node(self, name, factory, directory=None, create=1):
        if False:
            i = 10
            return i + 15
        ' Create node, and set it up to factory settings. '
        import SCons.Util
        node = factory(name, directory, create)
        node.set_noclean(self.noclean)
        node.set_precious(self.precious)
        if self.nodefault:
            self.env.Ignore('.', node)
        if self.alias:
            self.env.AlwaysBuild(self.env.Alias(self.alias, node))
        return node

    def Entry(self, name, directory=None, create=1):
        if False:
            return 10
        ' Create `SCons.Node.FS.Entry` '
        return self._create_node(name, self.env.fs.Entry, directory, create)

    def File(self, name, directory=None, create=1):
        if False:
            return 10
        ' Create `SCons.Node.FS.File` '
        return self._create_node(name, self.env.fs.File, directory, create)
_re_comment = re.compile('(#[^\\n\\r]+)$', re.M)
_re_lang = re.compile('([a-zA-Z0-9_]+)', re.M)

def _read_linguas_from_files(env, linguas_files=None):
    if False:
        return 10
    ' Parse `LINGUAS` file and return list of extracted languages '
    import SCons.Util
    import SCons.Environment
    global _re_comment
    global _re_lang
    if not SCons.Util.is_List(linguas_files) and (not SCons.Util.is_String(linguas_files)) and (not isinstance(linguas_files, SCons.Node.FS.Base)) and linguas_files:
        linguas_files = ['LINGUAS']
    if linguas_files is None:
        return []
    fnodes = env.arg2nodes(linguas_files)
    linguas = []
    for fnode in fnodes:
        contents = _re_comment.sub('', fnode.get_text_contents())
        ls = [l for l in _re_lang.findall(contents) if l]
        linguas.extend(ls)
    return linguas
from SCons.Builder import BuilderBase

class _POFileBuilder(BuilderBase):
    """ `PO` file builder.

    This is multi-target single-source builder. In typical situation the source
    is single `POT` file, e.g. `messages.pot`, and there are multiple `PO`
    targets to be updated from this `POT`. We must run
    `SCons.Builder.BuilderBase._execute()` separatelly for each target to track
    dependencies separatelly for each target file.

    **NOTE**: if we call `SCons.Builder.BuilderBase._execute(.., target, ...)`
    with target being list of all targets, all targets would be rebuilt each time
    one of the targets from this list is missing. This would happen, for example,
    when new language `ll` enters `LINGUAS_FILE` (at this moment there is no
    `ll.po` file yet). To avoid this, we override
    `SCons.Builder.BuilerBase._execute()` and call it separatelly for each
    target. Here we also append to the target list the languages read from
    `LINGUAS_FILE`.
    """

    def __init__(self, env, **kw):
        if False:
            i = 10
            return i + 15
        if 'suffix' not in kw:
            kw['suffix'] = '$POSUFFIX'
        if 'src_suffix' not in kw:
            kw['src_suffix'] = '$POTSUFFIX'
        if 'src_builder' not in kw:
            kw['src_builder'] = '_POTUpdateBuilder'
        if 'single_source' not in kw:
            kw['single_source'] = True
        alias = None
        if 'target_alias' in kw:
            alias = kw['target_alias']
            del kw['target_alias']
        if 'target_factory' not in kw:
            kw['target_factory'] = _POTargetFactory(env, alias=alias).File
        BuilderBase.__init__(self, **kw)

    def _execute(self, env, target, source, *args, **kw):
        if False:
            while True:
                i = 10
        " Execute builder's actions.\n\n        Here we append to `target` the languages read from `$LINGUAS_FILE` and\n        apply `SCons.Builder.BuilderBase._execute()` separatelly to each target.\n        The arguments and return value are same as for\n        `SCons.Builder.BuilderBase._execute()`.\n        "
        import SCons.Util
        import SCons.Node
        linguas_files = None
        if 'LINGUAS_FILE' in env and env['LINGUAS_FILE']:
            linguas_files = env['LINGUAS_FILE']
            env['LINGUAS_FILE'] = None
            linguas = _read_linguas_from_files(env, linguas_files)
            if SCons.Util.is_List(target):
                target.extend(linguas)
            elif target is not None:
                target = [target] + linguas
            else:
                target = linguas
        if not target:
            return BuilderBase._execute(self, env, target, source, *args, **kw)
        if not SCons.Util.is_List(target):
            target = [target]
        result = []
        for tgt in target:
            r = BuilderBase._execute(self, env, [tgt], source, *args, **kw)
            result.extend(r)
        if linguas_files is not None:
            env['LINGUAS_FILE'] = linguas_files
        return SCons.Node.NodeList(result)
import SCons.Environment

def _translate(env, target=None, source=SCons.Environment._null, *args, **kw):
    if False:
        for i in range(10):
            print('nop')
    ' Function for `Translate()` pseudo-builder '
    if target is None:
        target = []
    pot = env.POTUpdate(None, source, *args, **kw)
    po = env.POUpdate(target, pot, *args, **kw)
    return po

class RPaths(object):
    """ Callable object, which returns pathnames relative to SCons current
    working directory.

    It seems like `SCons.Node.FS.Base.get_path()` returns absolute paths
    for nodes that are outside of current working directory (`env.fs.getcwd()`).
    Here, we often have `SConscript`, `POT` and `PO` files within `po/`
    directory and source files (e.g. `*.c`) outside of it. When generating `POT`
    template file, references to source files are written to `POT` template, so
    a translator may later quickly jump to appropriate source file and line from
    its `PO` editor (e.g. `poedit`).  Relative paths in  `PO` file are usually
    interpreted by `PO` editor as paths relative to the place, where `PO` file
    lives. The absolute paths would make resultant `POT` file nonportable, as
    the references would be correct only on the machine, where `POT` file was
    recently re-created. For such reason, we need a function, which always
    returns relative paths. This is the purpose of `RPaths` callable object.

    The `__call__` method returns paths relative to current working directory, but
    we assume, that *xgettext(1)* is run from the directory, where target file is
    going to be created.

    Note, that this may not work for files distributed over several hosts or
    across different drives on windows. We assume here, that single local
    filesystem holds both source files and target `POT` templates.

    Intended use of `RPaths` - in `xgettext.py`::

      def generate(env):
          from GettextCommon import RPaths
          ...
          sources = '$( ${_concat( "", SOURCES, "", __env__, XgettextRPaths, TARGET, SOURCES)} $)'
          env.Append(
            ...
            XGETTEXTCOM = 'XGETTEXT ... ' + sources,
            ...
            XgettextRPaths = RPaths(env)
          )
    """

    def __init__(self, env):
        if False:
            for i in range(10):
                print('nop')
        ' Initialize `RPaths` callable object.\n\n          **Arguments**:\n\n            - *env* - a `SCons.Environment.Environment` object, defines *current\n              working dir*.\n        '
        self.env = env

    def __call__(self, nodes, *args, **kw):
        if False:
            i = 10
            return i + 15
        " Return nodes' paths (strings) relative to current working directory.\n\n          **Arguments**:\n\n            - *nodes* ([`SCons.Node.FS.Base`]) - list of nodes.\n            - *args* -  currently unused.\n            - *kw* - currently unused.\n\n          **Returns**:\n\n           - Tuple of strings, which represent paths relative to current working\n             directory (for given environment).\n        "
        import os
        import SCons.Node.FS
        rpaths = ()
        cwd = self.env.fs.getcwd().get_abspath()
        for node in nodes:
            rpath = None
            if isinstance(node, SCons.Node.FS.Base):
                rpath = os.path.relpath(node.get_abspath(), cwd)
            if rpath is not None:
                rpaths += (rpath,)
        return rpaths

def _init_po_files(target, source, env):
    if False:
        i = 10
        return i + 15
    ' Action function for `POInit` builder. '
    nop = lambda target, source, env: 0
    if 'POAUTOINIT' in env:
        autoinit = env['POAUTOINIT']
    else:
        autoinit = False
    for tgt in target:
        if not tgt.exists():
            if autoinit:
                action = SCons.Action.Action('$MSGINITCOM', '$MSGINITCOMSTR')
            else:
                msg = 'File ' + repr(str(tgt)) + ' does not exist. ' + 'If you are a translator, you can create it through: \n' + '$MSGINITCOM'
                action = SCons.Action.Action(nop, msg)
            status = action([tgt], source, env)
            if status:
                return status
    return 0

def _detect_xgettext(env):
    if False:
        for i in range(10):
            print('nop')
    ' Detects *xgettext(1)* binary '
    if 'XGETTEXT' in env:
        return env['XGETTEXT']
    xgettext = env.Detect('xgettext')
    if xgettext:
        return xgettext
    raise SCons.Errors.StopError(XgettextNotFound, 'Could not detect xgettext')
    return None

def _xgettext_exists(env):
    if False:
        print('Hello World!')
    return _detect_xgettext(env)

def _detect_msginit(env):
    if False:
        print('Hello World!')
    ' Detects *msginit(1)* program. '
    if 'MSGINIT' in env:
        return env['MSGINIT']
    msginit = env.Detect('msginit')
    if msginit:
        return msginit
    raise SCons.Errors.StopError(MsginitNotFound, 'Could not detect msginit')
    return None

def _msginit_exists(env):
    if False:
        while True:
            i = 10
    return _detect_msginit(env)

def _detect_msgmerge(env):
    if False:
        for i in range(10):
            print('nop')
    ' Detects *msgmerge(1)* program. '
    if 'MSGMERGE' in env:
        return env['MSGMERGE']
    msgmerge = env.Detect('msgmerge')
    if msgmerge:
        return msgmerge
    raise SCons.Errors.StopError(MsgmergeNotFound, 'Could not detect msgmerge')
    return None

def _msgmerge_exists(env):
    if False:
        print('Hello World!')
    return _detect_msgmerge(env)

def _detect_msgfmt(env):
    if False:
        for i in range(10):
            print('nop')
    ' Detects *msgmfmt(1)* program. '
    if 'MSGFMT' in env:
        return env['MSGFMT']
    msgfmt = env.Detect('msgfmt')
    if msgfmt:
        return msgfmt
    raise SCons.Errors.StopError(MsgfmtNotFound, 'Could not detect msgfmt')
    return None

def _msgfmt_exists(env):
    if False:
        return 10
    return _detect_msgfmt(env)

def tool_list(platform, env):
    if False:
        i = 10
        return i + 15
    ' List tools that shall be generated by top-level `gettext` tool '
    return ['xgettext', 'msginit', 'msgmerge', 'msgfmt']