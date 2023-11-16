import sys
import bzrlib
from bzrlib import commands, tests
from bzrlib.tests import features
from bzrlib.plugins.bash_completion.bashcomp import *
import subprocess

class BashCompletionMixin(object):
    """Component for testing execution of a bash completion script."""
    _test_needs_features = [features.bash_feature]
    script = None

    def complete(self, words, cword=-1):
        if False:
            while True:
                i = 10
        'Perform a bash completion.\n\n        :param words: a list of words representing the current command.\n        :param cword: the current word to complete, defaults to the last one.\n        '
        if self.script is None:
            self.script = self.get_script()
        proc = subprocess.Popen([features.bash_feature.path, '--noprofile'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cword < 0:
            cword = len(words) + cword
        input = '%s\n' % self.script
        input += 'COMP_WORDS=( %s )\n' % ' '.join(["'" + w.replace("'", "'\\''") + "'" for w in words])
        input += 'COMP_CWORD=%d\n' % cword
        input += '%s\n' % getattr(self, 'script_name', '_bzr')
        input += 'echo ${#COMPREPLY[*]}\n'
        input += "IFS=$'\\n'\n"
        input += 'echo "${COMPREPLY[*]}"\n'
        (out, err) = proc.communicate(input)
        if '' != err:
            raise AssertionError('Unexpected error message:\n%s' % err)
        self.assertEqual('', err, 'No messages to standard error')
        lines = out.split('\n')
        nlines = int(lines[0])
        del lines[0]
        self.assertEqual('', lines[-1], 'Newline at end')
        del lines[-1]
        if nlines == 0 and len(lines) == 1 and (lines[0] == ''):
            del lines[0]
        self.assertEqual(nlines, len(lines), 'No newlines in generated words')
        self.completion_result = set(lines)
        return self.completion_result

    def assertCompletionEquals(self, *words):
        if False:
            i = 10
            return i + 15
        self.assertEqual(set(words), self.completion_result)

    def assertCompletionContains(self, *words):
        if False:
            return 10
        missing = set(words) - self.completion_result
        if missing:
            raise AssertionError('Completion should contain %r but it has %r' % (missing, self.completion_result))

    def assertCompletionOmits(self, *words):
        if False:
            print('Hello World!')
        surplus = set(words) & self.completion_result
        if surplus:
            raise AssertionError('Completion should omit %r but it has %r' % (surplus, res, self.completion_result))

    def get_script(self):
        if False:
            for i in range(10):
                print('nop')
        commands.install_bzr_command_hooks()
        dc = DataCollector()
        data = dc.collect()
        cg = BashCodeGen(data)
        res = cg.function()
        return res

class TestBashCompletion(tests.TestCase, BashCompletionMixin):
    """Test bash completions that don't execute bzr."""

    def test_simple_scipt(self):
        if False:
            print('Hello World!')
        'Ensure that the test harness works as expected'
        self.script = '\n_bzr() {\n    COMPREPLY=()\n    # add all words in reverse order, with some markup around them\n    for ((i = ${#COMP_WORDS[@]}; i > 0; --i)); do\n        COMPREPLY+=( "-${COMP_WORDS[i-1]}+" )\n    done\n    # and append the current word\n    COMPREPLY+=( "+${COMP_WORDS[COMP_CWORD]}-" )\n}\n'
        self.complete(['foo', '"bar', "'baz"], cword=1)
        self.assertCompletionEquals("-'baz+", '-"bar+', '-foo+', '+"bar-')

    def test_cmd_ini(self):
        if False:
            print('Hello World!')
        self.complete(['bzr', 'ini'])
        self.assertCompletionContains('init', 'init-repo', 'init-repository')
        self.assertCompletionOmits('commit')

    def test_init_opts(self):
        if False:
            i = 10
            return i + 15
        self.complete(['bzr', 'init', '-'])
        self.assertCompletionContains('-h', '--2a', '--format=2a')

    def test_global_opts(self):
        if False:
            print('Hello World!')
        self.complete(['bzr', '-', 'init'], cword=1)
        self.assertCompletionContains('--no-plugins', '--builtin')

    def test_commit_dashm(self):
        if False:
            print('Hello World!')
        self.complete(['bzr', 'commit', '-m'])
        self.assertCompletionEquals('-m')

    def test_status_negated(self):
        if False:
            i = 10
            return i + 15
        self.complete(['bzr', 'status', '--n'])
        self.assertCompletionContains('--no-versioned', '--no-verbose')

    def test_init_format_any(self):
        if False:
            while True:
                i = 10
        self.complete(['bzr', 'init', '--format', '=', 'directory'], cword=3)
        self.assertCompletionContains('1.9', '2a')

    def test_init_format_2(self):
        if False:
            while True:
                i = 10
        self.complete(['bzr', 'init', '--format', '=', '2', 'directory'], cword=4)
        self.assertCompletionContains('2a')
        self.assertCompletionOmits('1.9')

class TestBashCompletionInvoking(tests.TestCaseWithTransport, BashCompletionMixin):
    """Test bash completions that might execute bzr.

    Only the syntax ``$(bzr ...`` is supported so far. The bzr command
    will be replaced by the bzr instance running this selftest.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestBashCompletionInvoking, self).setUp()
        if sys.platform == 'win32':
            raise tests.KnownFailure('see bug #709104, completion is broken on windows')

    def get_script(self):
        if False:
            while True:
                i = 10
        s = super(TestBashCompletionInvoking, self).get_script()
        return s.replace('$(bzr ', "$('%s' " % self.get_bzr_path())

    def test_revspec_tag_all(self):
        if False:
            while True:
                i = 10
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag1', 'null:')
        wt.branch.tags.set_tag('tag2', 'null:')
        wt.branch.tags.set_tag('3tag', 'null:')
        self.complete(['bzr', 'log', '-r', 'tag', ':'])
        self.assertCompletionEquals('tag1', 'tag2', '3tag')

    def test_revspec_tag_prefix(self):
        if False:
            i = 10
            return i + 15
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag1', 'null:')
        wt.branch.tags.set_tag('tag2', 'null:')
        wt.branch.tags.set_tag('3tag', 'null:')
        self.complete(['bzr', 'log', '-r', 'tag', ':', 't'])
        self.assertCompletionEquals('tag1', 'tag2')

    def test_revspec_tag_spaces(self):
        if False:
            return 10
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag with spaces', 'null:')
        self.complete(['bzr', 'log', '-r', 'tag', ':', 't'])
        self.assertCompletionEquals('tag\\ with\\ spaces')
        self.complete(['bzr', 'log', '-r', '"tag:t'])
        self.assertCompletionEquals('tag:tag with spaces')
        self.complete(['bzr', 'log', '-r', "'tag:t"])
        self.assertCompletionEquals('tag:tag with spaces')

    def test_revspec_tag_endrange(self):
        if False:
            i = 10
            return i + 15
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag1', 'null:')
        wt.branch.tags.set_tag('tag2', 'null:')
        self.complete(['bzr', 'log', '-r', '3..tag', ':', 't'])
        self.assertCompletionEquals('tag1', 'tag2')
        self.complete(['bzr', 'log', '-r', '"3..tag:t'])
        self.assertCompletionEquals('3..tag:tag1', '3..tag:tag2')
        self.complete(['bzr', 'log', '-r', "'3..tag:t"])
        self.assertCompletionEquals('3..tag:tag1', '3..tag:tag2')

class TestBashCodeGen(tests.TestCase):

    def test_command_names(self):
        if False:
            return 10
        data = CompletionData()
        bar = CommandData('bar')
        bar.aliases.append('baz')
        data.commands.append(bar)
        data.commands.append(CommandData('foo'))
        cg = BashCodeGen(data)
        self.assertEqual('bar baz foo', cg.command_names())

    def test_debug_output(self):
        if False:
            for i in range(10):
                print('nop')
        data = CompletionData()
        self.assertEqual('', BashCodeGen(data, debug=False).debug_output())
        self.assertTrue(BashCodeGen(data, debug=True).debug_output())

    def test_bzr_version(self):
        if False:
            for i in range(10):
                print('nop')
        data = CompletionData()
        cg = BashCodeGen(data)
        self.assertEqual('%s.' % bzrlib.version_string, cg.bzr_version())
        data.plugins['foo'] = PluginData('foo', '1.0')
        data.plugins['bar'] = PluginData('bar', '2.0')
        cg = BashCodeGen(data)
        self.assertEqual('%s and the following plugins:\n# bar 2.0\n# foo 1.0' % bzrlib.version_string, cg.bzr_version())

    def test_global_options(self):
        if False:
            return 10
        data = CompletionData()
        data.global_options.add('--foo')
        data.global_options.add('--bar')
        cg = BashCodeGen(data)
        self.assertEqual('--bar --foo', cg.global_options())

    def test_command_cases(self):
        if False:
            print('Hello World!')
        data = CompletionData()
        bar = CommandData('bar')
        bar.aliases.append('baz')
        bar.options.append(OptionData('--opt'))
        data.commands.append(bar)
        data.commands.append(CommandData('foo'))
        cg = BashCodeGen(data)
        self.assertEqualDiff('\tbar|baz)\n\t\tcmdOpts=( --opt )\n\t\t;;\n\tfoo)\n\t\tcmdOpts=(  )\n\t\t;;\n', cg.command_cases())

    def test_command_case(self):
        if False:
            i = 10
            return i + 15
        cmd = CommandData('cmd')
        cmd.plugin = PluginData('plugger', '1.0')
        bar = OptionData('--bar')
        bar.registry_keys = ['that', 'this']
        bar.error_messages.append('Some error message')
        cmd.options.append(bar)
        cmd.options.append(OptionData('--foo'))
        data = CompletionData()
        data.commands.append(cmd)
        cg = BashCodeGen(data)
        self.assertEqualDiff('\tcmd)\n\t\t# plugin "plugger 1.0"\n\t\t# Some error message\n\t\tcmdOpts=( --bar=that --bar=this --foo )\n\t\tcase $curOpt in\n\t\t\t--bar) optEnums=( that this ) ;;\n\t\tesac\n\t\t;;\n', cg.command_case(cmd))

class TestDataCollector(tests.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestDataCollector, self).setUp()
        commands.install_bzr_command_hooks()

    def test_global_options(self):
        if False:
            return 10
        dc = DataCollector()
        dc.global_options()
        self.assertSubset(['--no-plugins', '--builtin'], dc.data.global_options)

    def test_commands(self):
        if False:
            return 10
        dc = DataCollector()
        dc.commands()
        self.assertSubset(['init', 'init-repo', 'init-repository'], dc.data.all_command_aliases())

    def test_commands_from_plugins(self):
        if False:
            while True:
                i = 10
        dc = DataCollector()
        dc.commands()
        self.assertSubset(['bash-completion'], dc.data.all_command_aliases())

    def test_commit_dashm(self):
        if False:
            print('Hello World!')
        dc = DataCollector()
        cmd = dc.command('commit')
        self.assertSubset(['-m'], [str(o) for o in cmd.options])

    def test_status_negated(self):
        if False:
            print('Hello World!')
        dc = DataCollector()
        cmd = dc.command('status')
        self.assertSubset(['--no-versioned', '--no-verbose'], [str(o) for o in cmd.options])

    def test_init_format(self):
        if False:
            while True:
                i = 10
        dc = DataCollector()
        cmd = dc.command('init')
        for opt in cmd.options:
            if opt.name == '--format':
                self.assertSubset(['2a'], opt.registry_keys)
                return
        raise AssertionError('Option --format not found')

class BlackboxTests(tests.TestCase):

    def test_bash_completion(self):
        if False:
            i = 10
            return i + 15
        self.run_bzr('bash-completion')