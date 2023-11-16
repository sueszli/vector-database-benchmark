import io
import os
import pathlib
import signal
import tempfile
import shutil
import sys
import random
import filecmp
from os.path import splitext
from tempfile import NamedTemporaryFile
import ninja
import hashlib
import SCons
from SCons.Script import COMMAND_LINE_TARGETS
from SCons.Util import wait_for_process_to_die
from SCons.Errors import InternalError
from .Globals import COMMAND_TYPES, NINJA_RULES, NINJA_POOLS, NINJA_CUSTOM_HANDLERS, NINJA_DEFAULT_TARGETS
from .Rules import _install_action_function, _mkdir_action_function, _lib_symlink_action_function, _copy_action_function
from .Utils import get_path, alias_to_ninja_build, generate_depfile, ninja_noop, get_order_only, get_outputs, get_inputs, get_dependencies, get_rule, get_command_env, to_escaped_list, ninja_sorted_build
from .Methods import get_command

class NinjaState:
    """Maintains state of Ninja build system as it's translated from SCons."""

    def __init__(self, env, ninja_file, ninja_syntax):
        if False:
            return 10
        self.env = env
        self.ninja_file = ninja_file
        self.ninja_bin_path = env.get('NINJA')
        if not self.ninja_bin_path:
            ninja_bin = 'ninja.exe' if env['PLATFORM'] == 'win32' else 'ninja'
            self.ninja_bin_path = os.path.abspath(os.path.join(ninja.__file__, os.pardir, 'data', 'bin', ninja_bin))
            if not os.path.exists(self.ninja_bin_path):
                self.ninja_bin_path = ninja_bin
        self.ninja_syntax = ninja_syntax
        self.writer_class = ninja_syntax.Writer
        self.__generated = False
        self.translator = SConsToNinjaTranslator(env)
        self.generated_suffixes = env.get('NINJA_GENERATED_SOURCE_SUFFIXES', [])
        self.builds = dict()
        scons_escape = env.get('ESCAPE', lambda x: x)
        scons_daemon_port = None
        os.makedirs(get_path(self.env.get('NINJA_DIR')), exist_ok=True)
        scons_daemon_port_file = str(pathlib.Path(get_path(self.env.get('NINJA_DIR'))) / 'scons_daemon_portfile')
        if env.get('NINJA_SCONS_DAEMON_PORT') is not None:
            scons_daemon_port = int(env.get('NINJA_SCONS_DAEMON_PORT'))
        elif os.path.exists(scons_daemon_port_file):
            with open(scons_daemon_port_file) as f:
                scons_daemon_port = int(f.read())
        else:
            scons_daemon_port = random.randint(10000, 60000)
        with open(scons_daemon_port_file, 'w') as f:
            f.write(str(scons_daemon_port))
        python_bin = ''
        if os.path.basename(sys.argv[0]) == 'scons.py':
            python_bin = ninja_syntax.escape(scons_escape(sys.executable))
        self.variables = {'COPY': 'cmd.exe /c 1>NUL copy' if sys.platform == 'win32' else 'cp', 'PORT': scons_daemon_port, 'NINJA_DIR_PATH': env.get('NINJA_DIR').abspath, 'PYTHON_BIN': sys.executable, 'NINJA_TOOL_DIR': pathlib.Path(__file__).parent, 'NINJA_SCONS_DAEMON_KEEP_ALIVE': str(env.get('NINJA_SCONS_DAEMON_KEEP_ALIVE')), 'SCONS_INVOCATION': '{} {} --disable-ninja __NINJA_NO=1 $out'.format(python_bin, ' '.join([ninja_syntax.escape(scons_escape(arg)) for arg in sys.argv if arg not in COMMAND_LINE_TARGETS])), 'SCONS_INVOCATION_W_TARGETS': '{} {} NINJA_DISABLE_AUTO_RUN=1'.format(python_bin, ' '.join([ninja_syntax.escape(scons_escape(arg)) for arg in sys.argv if arg != 'NINJA_DISABLE_AUTO_RUN=1'])), 'msvc_deps_prefix': env.get('NINJA_MSVC_DEPS_PREFIX', 'Note: including file:')}
        self.rules = {'CMD': {'command': 'cmd /c $env$cmd $in $out' if sys.platform == 'win32' else '$env$cmd $in $out', 'description': 'Building $out', 'pool': 'local_pool'}, 'GENERATED_CMD': {'command': 'cmd /c $env$cmd' if sys.platform == 'win32' else '$env$cmd', 'description': 'Building $out', 'pool': 'local_pool'}, 'CC_RSP': {'command': '$env$CC @$out.rsp', 'description': 'Compiling $out', 'rspfile': '$out.rsp', 'rspfile_content': '$rspc'}, 'CXX_RSP': {'command': '$env$CXX @$out.rsp', 'description': 'Compiling $out', 'rspfile': '$out.rsp', 'rspfile_content': '$rspc'}, 'LINK_RSP': {'command': '$env$LINK @$out.rsp', 'description': 'Linking $out', 'rspfile': '$out.rsp', 'rspfile_content': '$rspc', 'pool': 'local_pool'}, 'AR_RSP': {'command': '{}$env$AR @$out.rsp'.format('' if sys.platform == 'win32' else 'rm -f $out && '), 'description': 'Archiving $out', 'rspfile': '$out.rsp', 'rspfile_content': '$rspc', 'pool': 'local_pool'}, 'CC': {'command': '$env$CC $rspc', 'description': 'Compiling $out'}, 'CXX': {'command': '$env$CXX $rspc', 'description': 'Compiling $out'}, 'LINK': {'command': '$env$LINK $rspc', 'description': 'Linking $out', 'pool': 'local_pool'}, 'AR': {'command': '{}$env$AR $rspc'.format('' if sys.platform == 'win32' else 'rm -f $out && '), 'description': 'Archiving $out', 'pool': 'local_pool'}, 'SYMLINK': {'command': 'cmd /c mklink $out $in' if sys.platform == 'win32' else 'ln -s $in $out', 'description': 'Symlink $in -> $out'}, 'INSTALL': {'command': '$COPY $in $out', 'description': 'Install $out', 'pool': 'install_pool', 'restat': 1}, 'TEMPLATE': {'command': '$PYTHON_BIN $NINJA_TOOL_DIR/ninja_daemon_build.py $PORT $NINJA_DIR_PATH $out', 'description': 'Defer to SCons to build $out', 'pool': 'local_pool', 'restat': 1}, 'EXIT_SCONS_DAEMON': {'command': '$PYTHON_BIN $NINJA_TOOL_DIR/ninja_daemon_build.py $PORT $NINJA_DIR_PATH --exit', 'description': 'Shutting down ninja scons daemon server', 'pool': 'local_pool', 'restat': 1}, 'SCONS': {'command': '$SCONS_INVOCATION $out', 'description': '$SCONS_INVOCATION $out', 'pool': 'scons_pool', 'restat': 1}, 'SCONS_DAEMON': {'command': '$PYTHON_BIN $NINJA_TOOL_DIR/ninja_run_daemon.py $PORT $NINJA_DIR_PATH $NINJA_SCONS_DAEMON_KEEP_ALIVE $SCONS_INVOCATION', 'description': 'Starting scons daemon...', 'pool': 'local_pool', 'restat': 1}, 'REGENERATE': {'command': '$SCONS_INVOCATION_W_TARGETS', 'description': 'Regenerating $self', 'generator': 1, 'pool': 'console', 'restat': 1}}
        if env['PLATFORM'] == 'darwin' and env.get('AR', '') == 'ar':
            self.rules['AR'] = {'command': 'rm -f $out && $env$AR $rspc', 'description': 'Archiving $out', 'pool': 'local_pool'}
        self.pools = {'scons_pool': 1}

    def add_build(self, node):
        if False:
            for i in range(10):
                print('nop')
        if not node.has_builder():
            return False
        if isinstance(node, SCons.Node.Python.Value):
            return False
        if isinstance(node, SCons.Node.Alias.Alias):
            build = alias_to_ninja_build(node)
        else:
            build = self.translator.action_to_ninja_build(node)
        if build is None:
            return False
        node_string = str(node)
        if node_string in self.builds:
            warn_msg = f'Alias {node_string} name the same as File node, ninja does not support this. Renaming Alias {node_string} to {node_string}_alias.'
            if isinstance(node, SCons.Node.Alias.Alias):
                for (i, output) in enumerate(build['outputs']):
                    if output == node_string:
                        build['outputs'][i] += '_alias'
                node_string += '_alias'
                print(warn_msg)
            elif self.builds[node_string]['rule'] == 'phony':
                for (i, output) in enumerate(self.builds[node_string]['outputs']):
                    if output == node_string:
                        self.builds[node_string]['outputs'][i] += '_alias'
                tmp_build = self.builds[node_string].copy()
                del self.builds[node_string]
                node_string += '_alias'
                self.builds[node_string] = tmp_build
                print(warn_msg)
            else:
                raise InternalError('Node {} added to ninja build state more than once'.format(node_string))
        self.builds[node_string] = build
        return True

    def is_generated_source(self, output):
        if False:
            print('Hello World!')
        'Check if output ends with a known generated suffix.'
        (_, suffix) = splitext(output)
        return suffix in self.generated_suffixes

    def has_generated_sources(self, output):
        if False:
            print('Hello World!')
        '\n        Determine if output indicates this is a generated header file.\n        '
        for generated in output:
            if self.is_generated_source(generated):
                return True
        return False

    def generate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate the build.ninja.\n\n        This should only be called once for the lifetime of this object.\n        '
        if self.__generated:
            return
        num_jobs = self.env.get('NINJA_MAX_JOBS', self.env.GetOption('num_jobs'))
        self.pools.update({'local_pool': num_jobs, 'install_pool': num_jobs / 2})
        deps_format = self.env.get('NINJA_DEPFILE_PARSE_FORMAT', 'msvc' if self.env['PLATFORM'] == 'win32' else 'gcc')
        for rule in ['CC', 'CXX']:
            if deps_format == 'msvc':
                self.rules[rule]['deps'] = 'msvc'
            elif deps_format == 'gcc' or deps_format == 'clang':
                self.rules[rule]['deps'] = 'gcc'
                self.rules[rule]['depfile'] = '$out.d'
            else:
                raise Exception(f"Unknown 'NINJA_DEPFILE_PARSE_FORMAT'={self.env['NINJA_DEPFILE_PARSE_FORMAT']}, use 'mvsc', 'gcc', or 'clang'.")
        for (key, rule) in self.env.get(NINJA_RULES, {}).items():
            if rule.get('rspfile') is not None:
                self.rules.update({key + '_RSP': rule})
                non_rsp_rule = rule.copy()
                del non_rsp_rule['rspfile']
                del non_rsp_rule['rspfile_content']
                self.rules.update({key: non_rsp_rule})
            else:
                self.rules.update({key: rule})
        self.pools.update(self.env.get(NINJA_POOLS, {}))
        content = io.StringIO()
        ninja = self.writer_class(content, width=100)
        ninja.comment('Generated by scons. DO NOT EDIT.')
        ninja.variable('builddir', get_path(self.env.Dir(self.env['NINJA_DIR']).path))
        for (pool_name, size) in sorted(self.pools.items()):
            ninja.pool(pool_name, min(self.env.get('NINJA_MAX_JOBS', size), size))
        for (var, val) in sorted(self.variables.items()):
            ninja.variable(var, val)
        for (rule, kwargs) in sorted(self.rules.items()):
            if self.env.get('NINJA_MAX_JOBS') is not None and 'pool' not in kwargs:
                kwargs['pool'] = 'local_pool'
            ninja.rule(rule, **kwargs)
        generated_sources_alias = self.env.get('NINJA_GENERATED_SOURCE_ALIAS_NAME')
        generated_sources_build = None
        if generated_sources_alias:
            generated_sources_build = self.builds.get(generated_sources_alias)
            if generated_sources_build is None or generated_sources_build['rule'] != 'phony':
                raise Exception("ERROR: 'NINJA_GENERATED_SOURCE_ALIAS_NAME' set, but no matching Alias object found.")
        if generated_sources_alias and generated_sources_build:
            generated_source_files = sorted([] if not generated_sources_build else generated_sources_build['implicit'])

            def check_generated_source_deps(build):
                if False:
                    for i in range(10):
                        print('nop')
                return build != generated_sources_build and set(build['outputs']).isdisjoint(generated_source_files)
        else:
            generated_sources_build = None
            generated_source_files = sorted({output for build in self.builds.values() if self.has_generated_sources(build['outputs']) for output in build['outputs'] if self.is_generated_source(output)})
            if generated_source_files:
                generated_sources_alias = '_ninja_generated_sources'
                ninja.build(outputs=generated_sources_alias, rule='phony', implicit=generated_source_files)

                def check_generated_source_deps(build):
                    if False:
                        while True:
                            i = 10
                    return not build['rule'] == 'INSTALL' and set(build['outputs']).isdisjoint(generated_source_files) and set(build.get('implicit', [])).isdisjoint(generated_source_files)
        template_builders = []
        scons_compiledb = False
        if SCons.Script._Get_Default_Targets == SCons.Script._Set_Default_Targets_Has_Not_Been_Called:
            all_targets = set()
        else:
            all_targets = None
        for build in [self.builds[key] for key in sorted(self.builds.keys())]:
            if 'compile_commands.json' in build['outputs']:
                scons_compiledb = True
            if all_targets is not None and build['rule'] != 'phony':
                all_targets = all_targets | set(build['outputs'])
            if build['rule'] == 'TEMPLATE':
                template_builders.append(build)
                continue
            if 'implicit' in build:
                build['implicit'].sort()
            if generated_source_files and check_generated_source_deps(build):
                order_only = build.get('order_only', [])
                order_only.append(generated_sources_alias)
                build['order_only'] = order_only
            if 'order_only' in build:
                build['order_only'].sort()
            rule = self.rules.get(build['rule'])
            if rule is not None and (rule.get('deps') or rule.get('rspfile')):
                (first_output, remaining_outputs) = (build['outputs'][0], build['outputs'][1:])
                if remaining_outputs:
                    ninja_sorted_build(ninja, outputs=remaining_outputs, rule='phony', implicit=first_output)
                build['outputs'] = first_output
            if rule is not None and rule.get('depfile') and build.get('deps_files'):
                path = build['outputs'] if SCons.Util.is_List(build['outputs']) else [build['outputs']]
                generate_depfile(self.env, path[0], build.pop('deps_files', []))
            if 'inputs' in build:
                build['inputs'].sort()
            ninja_sorted_build(ninja, **build)
        scons_daemon_dirty = str(pathlib.Path(get_path(self.env.get('NINJA_DIR'))) / 'scons_daemon_dirty')
        for template_builder in template_builders:
            template_builder['implicit'] += [scons_daemon_dirty]
            ninja_sorted_build(ninja, **template_builder)
        ninja_file_path = self.env.File(self.ninja_file).path
        regenerate_deps = to_escaped_list(self.env, self.env['NINJA_REGENERATE_DEPS'])
        ninja_sorted_build(ninja, outputs=ninja_file_path, rule='REGENERATE', implicit=regenerate_deps, variables={'self': ninja_file_path})
        ninja_sorted_build(ninja, outputs=regenerate_deps, rule='phony', variables={'self': ninja_file_path})
        if not scons_compiledb:
            ninja_sorted_build(ninja, outputs='compile_commands.json', rule='CMD', pool='console', implicit=[str(self.ninja_file)], variables={'cmd': '{} -f {} -t compdb {}CC CXX > compile_commands.json'.format(self.ninja_bin_path, str(self.ninja_file), '-x ' if self.env.get('NINJA_COMPDB_EXPAND', True) else '')})
            ninja_sorted_build(ninja, outputs='compiledb', rule='phony', implicit=['compile_commands.json'])
        ninja_sorted_build(ninja, outputs=['run_ninja_scons_daemon_phony', scons_daemon_dirty], rule='SCONS_DAEMON')
        ninja.build('shutdown_ninja_scons_daemon_phony', rule='EXIT_SCONS_DAEMON')
        if all_targets is None:
            all_targets = [str(node) for node in NINJA_DEFAULT_TARGETS]
        else:
            all_targets = list(all_targets)
        if len(all_targets) == 0:
            all_targets = ['phony_default']
            ninja_sorted_build(ninja, outputs=all_targets, rule='phony')
        ninja.default([self.ninja_syntax.escape_path(path) for path in sorted(all_targets)])
        with NamedTemporaryFile(delete=False, mode='w') as temp_ninja_file:
            temp_ninja_file.write(content.getvalue())
        if self.env.GetOption('skip_ninja_regen') and os.path.exists(ninja_file_path) and filecmp.cmp(temp_ninja_file.name, ninja_file_path):
            os.unlink(temp_ninja_file.name)
        else:
            daemon_dir = pathlib.Path(tempfile.gettempdir()) / ('scons_daemon_' + str(hashlib.md5(str(get_path(self.env['NINJA_DIR'])).encode()).hexdigest()))
            pidfile = None
            if os.path.exists(scons_daemon_dirty):
                pidfile = scons_daemon_dirty
            elif os.path.exists(daemon_dir / 'pidfile'):
                pidfile = daemon_dir / 'pidfile'
            if pidfile:
                with open(pidfile) as f:
                    pid = int(f.readline())
                    try:
                        os.kill(pid, signal.SIGINT)
                    except OSError:
                        pass
                wait_for_process_to_die(pid)
            if os.path.exists(scons_daemon_dirty):
                os.unlink(scons_daemon_dirty)
            shutil.move(temp_ninja_file.name, ninja_file_path)
        self.__generated = True

class SConsToNinjaTranslator:
    """Translates SCons Actions into Ninja build objects."""

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        self.env = env
        self.func_handlers = {'_createSource': ninja_noop, 'SharedFlagChecker': ninja_noop, 'installFunc': _install_action_function, 'MkdirFunc': _mkdir_action_function, 'Mkdir': _mkdir_action_function, 'LibSymlinksActionFunction': _lib_symlink_action_function, 'Copy': _copy_action_function}
        self.loaded_custom = False

    def action_to_ninja_build(self, node, action=None):
        if False:
            while True:
                i = 10
        'Generate build arguments dictionary for node.'
        if not self.loaded_custom:
            self.func_handlers.update(self.env[NINJA_CUSTOM_HANDLERS])
            self.loaded_custom = True
        if node.builder is None:
            return None
        if action is None:
            action = node.builder.action
        if node.env and node.env.get('NINJA_SKIP'):
            return None
        build = {}
        env = node.env if node.env else self.env
        if SCons.Tool.ninja.NINJA_STATE.ninja_file == str(node):
            build = None
        elif isinstance(action, SCons.Action.FunctionAction):
            build = self.handle_func_action(node, action)
        elif isinstance(action, SCons.Action.LazyAction):
            action = action._generate_cache(env)
            build = self.action_to_ninja_build(node, action=action)
        elif isinstance(action, SCons.Action.ListAction):
            build = self.handle_list_action(node, action)
        elif isinstance(action, COMMAND_TYPES):
            build = get_command(env, node, action)
        else:
            return {'rule': 'TEMPLATE', 'order_only': get_order_only(node), 'outputs': get_outputs(node), 'inputs': get_inputs(node), 'implicit': get_dependencies(node, skip_sources=True)}
        if build is not None:
            build['order_only'] = get_order_only(node)
        if not node.is_conftest():
            node_callback = node.check_attributes('ninja_build_callback')
            if callable(node_callback):
                node_callback(env, node, build)
        return build

    def handle_func_action(self, node, action):
        if False:
            print('Hello World!')
        'Determine how to handle the function action.'
        name = action.function_name()
        if name == 'ninja_builder':
            return None
        handler = self.func_handlers.get(name, None)
        if handler is not None:
            return handler(node.env if node.env else self.env, node)
        elif name == 'ActionCaller':
            action_to_call = str(action).split('(')[0].strip()
            handler = self.func_handlers.get(action_to_call, None)
            if handler is not None:
                return handler(node.env if node.env else self.env, node)
        SCons.Warnings.SConsWarning('Found unhandled function action {},  generating scons command to build\nNote: this is less efficient than Ninja, you can write your own ninja build generator for this function using NinjaRegisterFunctionHandler'.format(name))
        return {'rule': 'TEMPLATE', 'order_only': get_order_only(node), 'outputs': get_outputs(node), 'inputs': get_inputs(node), 'implicit': get_dependencies(node, skip_sources=True)}

    def handle_list_action(self, node, action):
        if False:
            while True:
                i = 10
        'TODO write this comment'
        results = [self.action_to_ninja_build(node, action=act) for act in action.list if act is not None]
        results = [result for result in results if result is not None and result['outputs']]
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        all_outputs = list({output for build in results for output in build['outputs']})
        dependencies = list({dep for build in results for dep in build.get('implicit', [])})
        if results[0]['rule'] == 'CMD' or results[0]['rule'] == 'GENERATED_CMD':
            cmdline = ''
            for cmd in results:
                if not cmd.get('variables') or not cmd['variables'].get('cmd'):
                    continue
                cmdstr = cmd['variables']['cmd'].strip()
                if not cmdstr:
                    continue
                if cmdstr in cmdline:
                    continue
                if cmdline:
                    cmdline += ' && '
                cmdline += cmdstr
            cmdline = cmdline.strip()
            env = node.env if node.env else self.env
            executor = node.get_executor()
            if executor is not None:
                targets = executor.get_all_targets()
            elif hasattr(node, 'target_peers'):
                targets = node.target_peers
            else:
                targets = [node]
            if cmdline:
                ninja_build = {'outputs': all_outputs, 'rule': get_rule(node, 'GENERATED_CMD'), 'variables': {'cmd': cmdline, 'env': get_command_env(env, targets, node.sources)}, 'implicit': dependencies}
                if node.env and node.env.get('NINJA_POOL', None) is not None:
                    ninja_build['pool'] = node.env['pool']
                return ninja_build
        elif results[0]['rule'] == 'phony':
            return {'outputs': all_outputs, 'rule': 'phony', 'implicit': dependencies}
        elif results[0]['rule'] == 'INSTALL':
            return {'outputs': all_outputs, 'rule': get_rule(node, 'INSTALL'), 'inputs': get_inputs(node), 'implicit': dependencies}
        return {'rule': 'TEMPLATE', 'order_only': get_order_only(node), 'outputs': get_outputs(node), 'inputs': get_inputs(node), 'implicit': get_dependencies(node, skip_sources=True)}