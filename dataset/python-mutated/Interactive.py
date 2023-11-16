from __future__ import print_function
__revision__ = 'src/engine/SCons/Script/Interactive.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__doc__ = '\nSCons interactive mode\n'
import cmd
import copy
import os
import re
import shlex
import sys
try:
    import readline
except ImportError:
    pass

class SConsInteractiveCmd(cmd.Cmd):
    """
build [TARGETS]         Build the specified TARGETS and their dependencies. 'b' is a synonym.
clean [TARGETS]         Clean (remove) the specified TARGETS and their dependencies.  'c' is a synonym.
exit                    Exit SCons interactive mode.
help [COMMAND]          Prints help for the specified COMMAND.  'h' and '?' are synonyms.
shell [COMMANDLINE]     Execute COMMANDLINE in a subshell.  'sh' and '!' are synonyms.
version                 Prints SCons version information.
"""
    synonyms = {'b': 'build', 'c': 'clean', 'h': 'help', 'scons': 'build', 'sh': 'shell'}

    def __init__(self, **kw):
        if False:
            return 10
        cmd.Cmd.__init__(self)
        for (key, val) in kw.items():
            setattr(self, key, val)
        if sys.platform == 'win32':
            self.shell_variable = 'COMSPEC'
        else:
            self.shell_variable = 'SHELL'

    def default(self, argv):
        if False:
            for i in range(10):
                print('nop')
        print('*** Unknown command: %s' % argv[0])

    def onecmd(self, line):
        if False:
            for i in range(10):
                print('nop')
        line = line.strip()
        if not line:
            print(self.lastcmd)
            return self.emptyline()
        self.lastcmd = line
        if line[0] == '!':
            line = 'shell ' + line[1:]
        elif line[0] == '?':
            line = 'help ' + line[1:]
        if os.sep == '\\':
            line = line.replace('\\', '\\\\')
        argv = shlex.split(line)
        argv[0] = self.synonyms.get(argv[0], argv[0])
        if not argv[0]:
            return self.default(line)
        else:
            try:
                func = getattr(self, 'do_' + argv[0])
            except AttributeError:
                return self.default(argv)
            return func(argv)

    def do_build(self, argv):
        if False:
            i = 10
            return i + 15
        "        build [TARGETS]         Build the specified TARGETS and their\n                                dependencies.  'b' is a synonym.\n        "
        import SCons.Node
        import SCons.SConsign
        import SCons.Script.Main
        options = copy.deepcopy(self.options)
        (options, targets) = self.parser.parse_args(argv[1:], values=options)
        SCons.Script.COMMAND_LINE_TARGETS = targets
        if targets:
            SCons.Script.BUILD_TARGETS = targets
        else:
            SCons.Script.BUILD_TARGETS = SCons.Script._build_plus_default
        nodes = SCons.Script.Main._build_targets(self.fs, options, targets, self.target_top)
        if not nodes:
            return
        x = []
        for n in nodes:
            x.extend(n.alter_targets()[0])
        nodes.extend(x)
        SCons.Script.Main.progress_display('scons: Clearing cached node information ...')
        seen_nodes = {}

        def get_unseen_children(node, parent, seen_nodes=seen_nodes):
            if False:
                while True:
                    i = 10

            def is_unseen(node, seen_nodes=seen_nodes):
                if False:
                    i = 10
                    return i + 15
                return node not in seen_nodes
            return [child for child in node.children(scan=1) if is_unseen(child)]

        def add_to_seen_nodes(node, parent, seen_nodes=seen_nodes):
            if False:
                while True:
                    i = 10
            seen_nodes[node] = 1
            try:
                rfile_method = node.rfile
            except AttributeError:
                return
            else:
                rfile = rfile_method()
            if rfile != node:
                seen_nodes[rfile] = 1
        for node in nodes:
            walker = SCons.Node.Walker(node, kids_func=get_unseen_children, eval_func=add_to_seen_nodes)
            n = walker.get_next()
            while n:
                n = walker.get_next()
        for node in list(seen_nodes.keys()):
            node.clear()
            node.set_state(SCons.Node.no_state)
            node.implicit = None
        SCons.SConsign.Reset()
        SCons.Script.Main.progress_display('scons: done clearing node information.')

    def do_clean(self, argv):
        if False:
            i = 10
            return i + 15
        "        clean [TARGETS]         Clean (remove) the specified TARGETS\n                                and their dependencies.  'c' is a synonym.\n        "
        return self.do_build(['build', '--clean'] + argv[1:])

    def do_EOF(self, argv):
        if False:
            print('Hello World!')
        print()
        self.do_exit(argv)

    def _do_one_help(self, arg):
        if False:
            while True:
                i = 10
        try:
            func = getattr(self, 'help_' + arg)
        except AttributeError:
            try:
                func = getattr(self, 'do_' + arg)
            except AttributeError:
                doc = None
            else:
                doc = self._doc_to_help(func)
            if doc:
                sys.stdout.write(doc + '\n')
                sys.stdout.flush()
        else:
            doc = self.strip_initial_spaces(func())
            if doc:
                sys.stdout.write(doc + '\n')
                sys.stdout.flush()

    def _doc_to_help(self, obj):
        if False:
            while True:
                i = 10
        doc = obj.__doc__
        if doc is None:
            return ''
        return self._strip_initial_spaces(doc)

    def _strip_initial_spaces(self, s):
        if False:
            i = 10
            return i + 15
        lines = s.split('\n')
        spaces = re.match(' *', lines[0]).group(0)

        def strip_spaces(l, spaces=spaces):
            if False:
                for i in range(10):
                    print('nop')
            if l[:len(spaces)] == spaces:
                l = l[len(spaces):]
            return l
        lines = list(map(strip_spaces, lines))
        return '\n'.join(lines)

    def do_exit(self, argv):
        if False:
            for i in range(10):
                print('nop')
        '        exit                    Exit SCons interactive mode.\n        '
        sys.exit(0)

    def do_help(self, argv):
        if False:
            while True:
                i = 10
        "        help [COMMAND]          Prints help for the specified COMMAND.  'h'\n                                and '?' are synonyms.\n        "
        if argv[1:]:
            for arg in argv[1:]:
                if self._do_one_help(arg):
                    break
        else:
            doc = self._doc_to_help(self.__class__)
            if doc:
                sys.stdout.write(doc + '\n')
                sys.stdout.flush()

    def do_shell(self, argv):
        if False:
            i = 10
            return i + 15
        "        shell [COMMANDLINE]     Execute COMMANDLINE in a subshell.  'sh' and\n                                '!' are synonyms.\n        "
        import subprocess
        argv = argv[1:]
        if not argv:
            argv = os.environ[self.shell_variable]
        try:
            p = subprocess.Popen(argv, shell=sys.platform == 'win32')
        except EnvironmentError as e:
            sys.stderr.write('scons: %s: %s\n' % (argv[0], e.strerror))
        else:
            p.wait()

    def do_version(self, argv):
        if False:
            for i in range(10):
                print('nop')
        '        version                 Prints SCons version information.\n        '
        sys.stdout.write(self.parser.version + '\n')

def interact(fs, parser, options, targets, target_top):
    if False:
        while True:
            i = 10
    c = SConsInteractiveCmd(prompt='scons>>> ', fs=fs, parser=parser, options=options, targets=targets, target_top=target_top)
    c.cmdloop()