"""
Implements the ability for SCons to emit a compilation database for the MongoDB project. See
http://clang.llvm.org/docs/JSONCompilationDatabase.html for details on what a compilation
database is, and why you might want one. The only user visible entry point here is
'env.CompilationDatabase'. This method takes an optional 'target' to name the file that
should hold the compilation database, otherwise, the file defaults to compile_commands.json,
which is the name that most clang tools search for by default.
"""
import json
import itertools
import fnmatch
import SCons
from SCons.Platform import TempFileMunge
from .cxx import CXXSuffixes
from .cc import CSuffixes
from .asm import ASSuffixes, ASPPSuffixes
__COMPILATION_DB_ENTRIES = []

class __CompilationDbNode(SCons.Node.Python.Value):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        SCons.Node.Python.Value.__init__(self, value)
        self.Decider(changed_since_last_build_node)

def changed_since_last_build_node(child, target, prev_ni, node):
    if False:
        while True:
            i = 10
    ' Dummy decider to force always building'
    return True

def make_emit_compilation_DB_entry(comstr):
    if False:
        print('Hello World!')
    '\n    Effectively this creates a lambda function to capture:\n    * command line\n    * source\n    * target\n    :param comstr: unevaluated command line\n    :return: an emitter which has captured the above\n    '
    user_action = SCons.Action.Action(comstr)

    def emit_compilation_db_entry(target, source, env):
        if False:
            i = 10
            return i + 15
        '\n        This emitter will be added to each c/c++ object build to capture the info needed\n        for clang tools\n        :param target: target node(s)\n        :param source: source node(s)\n        :param env: Environment for use building this node\n        :return: target(s), source(s)\n        '
        dbtarget = __CompilationDbNode(source)
        entry = env.__COMPILATIONDB_Entry(target=dbtarget, source=[], __COMPILATIONDB_UOUTPUT=target, __COMPILATIONDB_USOURCE=source, __COMPILATIONDB_UACTION=user_action, __COMPILATIONDB_ENV=env)
        env.AlwaysBuild(entry)
        env.NoCache(entry)
        __COMPILATION_DB_ENTRIES.append(dbtarget)
        return (target, source)
    return emit_compilation_db_entry

class CompDBTEMPFILE(TempFileMunge):

    def __call__(self, target, source, env, for_signature):
        if False:
            return 10
        return self.cmd

def compilation_db_entry_action(target, source, env, **kw):
    if False:
        print('Hello World!')
    '\n    Create a dictionary with evaluated command line, target, source\n    and store that info as an attribute on the target\n    (Which has been stored in __COMPILATION_DB_ENTRIES array\n    :param target: target node(s)\n    :param source: source node(s)\n    :param env: Environment for use building this node\n    :param kw:\n    :return: None\n    '
    command = env['__COMPILATIONDB_UACTION'].strfunction(target=env['__COMPILATIONDB_UOUTPUT'], source=env['__COMPILATIONDB_USOURCE'], env=env['__COMPILATIONDB_ENV'], overrides={'TEMPFILE': CompDBTEMPFILE})
    entry = {'directory': env.Dir('#').abspath, 'command': command, 'file': env['__COMPILATIONDB_USOURCE'][0], 'output': env['__COMPILATIONDB_UOUTPUT'][0]}
    target[0].write(entry)

def write_compilation_db(target, source, env):
    if False:
        print('Hello World!')
    entries = []
    use_abspath = env['COMPILATIONDB_USE_ABSPATH'] in [True, 1, 'True', 'true']
    use_path_filter = env.subst('$COMPILATIONDB_PATH_FILTER')
    for s in __COMPILATION_DB_ENTRIES:
        entry = s.read()
        source_file = entry['file']
        output_file = entry['output']
        if use_abspath:
            source_file = source_file.srcnode().abspath
            output_file = output_file.abspath
        else:
            source_file = source_file.srcnode().path
            output_file = output_file.path
        if use_path_filter and (not fnmatch.fnmatch(output_file, use_path_filter)):
            continue
        path_entry = {'directory': entry['directory'], 'command': entry['command'], 'file': source_file, 'output': output_file}
        entries.append(path_entry)
    with open(target[0].path, 'w') as output_file:
        json.dump(entries, output_file, sort_keys=True, indent=4, separators=(',', ': '))
        output_file.write('\n')

def scan_compilation_db(node, env, path):
    if False:
        print('Hello World!')
    return __COMPILATION_DB_ENTRIES

def compilation_db_emitter(target, source, env):
    if False:
        i = 10
        return i + 15
    ' fix up the source/targets '
    if not target and len(source) == 1:
        target = source
    if not target:
        target = ['compile_commands.json']
    if source:
        source = []
    return (target, source)

def generate(env, **kwargs):
    if False:
        print('Hello World!')
    (static_obj, shared_obj) = SCons.Tool.createObjBuilders(env)
    env['COMPILATIONDB_COMSTR'] = kwargs.get('COMPILATIONDB_COMSTR', 'Building compilation database $TARGET')
    components_by_suffix = itertools.chain(itertools.product(CSuffixes, [(static_obj, SCons.Defaults.StaticObjectEmitter, '$CCCOM'), (shared_obj, SCons.Defaults.SharedObjectEmitter, '$SHCCCOM')]), itertools.product(CXXSuffixes, [(static_obj, SCons.Defaults.StaticObjectEmitter, '$CXXCOM'), (shared_obj, SCons.Defaults.SharedObjectEmitter, '$SHCXXCOM')]), itertools.product(ASSuffixes, [(static_obj, SCons.Defaults.StaticObjectEmitter, '$ASCOM')], [(shared_obj, SCons.Defaults.SharedObjectEmitter, '$ASCOM')]), itertools.product(ASPPSuffixes, [(static_obj, SCons.Defaults.StaticObjectEmitter, '$ASPPCOM')], [(shared_obj, SCons.Defaults.SharedObjectEmitter, '$ASPPCOM')]))
    for entry in components_by_suffix:
        suffix = entry[0]
        (builder, base_emitter, command) = entry[1]
        emitter = builder.emitter.get(suffix, False)
        if emitter:
            builder.emitter[suffix] = SCons.Builder.ListEmitter([emitter, make_emit_compilation_DB_entry(command)])
    env['BUILDERS']['__COMPILATIONDB_Entry'] = SCons.Builder.Builder(action=SCons.Action.Action(compilation_db_entry_action, None))
    env['BUILDERS']['CompilationDatabase'] = SCons.Builder.Builder(action=SCons.Action.Action(write_compilation_db, '$COMPILATIONDB_COMSTR'), target_scanner=SCons.Scanner.Scanner(function=scan_compilation_db, node_class=None), emitter=compilation_db_emitter, suffix='json')
    env['COMPILATIONDB_USE_ABSPATH'] = False
    env['COMPILATIONDB_PATH_FILTER'] = ''

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return True