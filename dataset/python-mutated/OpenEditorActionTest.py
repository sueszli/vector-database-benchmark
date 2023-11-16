import os
import logging
import subprocess
import tempfile
import unittest
from importlib import reload
import coalib.results.result_actions.OpenEditorAction
from coalib.results.Diff import Diff
from coalib.results.Result import Result
from coalib.results.result_actions.OpenEditorAction import OpenEditorAction
from coalib.results.result_actions.ApplyPatchAction import ApplyPatchAction
from coalib.settings.Section import Section, Setting

class OpenEditorActionTest(unittest.TestCase):

    @staticmethod
    def fake_edit(commands):
        if False:
            return 10
        filename = commands[1]
        with open(filename) as f:
            lines = f.readlines()
        del lines[1]
        with open(filename, 'w') as f:
            f.writelines(lines)

    def setUp(self):
        if False:
            while True:
                i = 10
        (fahandle, self.fa) = tempfile.mkstemp()
        os.close(fahandle)
        (fbhandle, self.fb) = tempfile.mkstemp()
        os.close(fbhandle)
        self.old_subprocess_call = subprocess.call
        self.old_environ = os.environ

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.fa)
        os.remove(self.fb)
        subprocess.call = self.old_subprocess_call
        os.environ = self.old_environ

    def test_apply(self):
        if False:
            for i in range(10):
                print('nop')
        file_dict = {self.fa: ['1\n', '2\n', '3\n'], self.fb: ['1\n', '2\n', '3\n'], 'f_c': ['1\n', '2\n', '3\n']}
        diff_dict = {self.fb: Diff(file_dict[self.fb])}
        diff_dict[self.fb].change_line(3, '3\n', '3_changed\n')
        current_file_dict = {filename: diff_dict[filename].modified if filename in diff_dict else file_dict[filename] for filename in (self.fa, self.fb)}
        for filename in current_file_dict:
            with open(filename, 'w') as handle:
                handle.writelines(current_file_dict[filename])
        expected_file_dict = {self.fa: ['1\n', '3\n'], self.fb: ['1\n', '3_changed\n'], 'f_c': ['1\n', '2\n', '3\n']}
        section = Section('')
        section.append(Setting('editor', 'vim'))
        uut = OpenEditorAction()
        subprocess.call = self.fake_edit
        diff_dict = uut.apply_from_section(Result.from_values('origin', 'msg', self.fa), file_dict, diff_dict, section)
        diff_dict = uut.apply_from_section(Result.from_values('origin', 'msg', self.fb), file_dict, diff_dict, section)
        for filename in diff_dict:
            file_dict[filename] = diff_dict[filename].modified
        self.assertEqual(file_dict, expected_file_dict)

    def test_apply_rename(self):
        if False:
            print('Hello World!')
        file_dict = {self.fa: ['1\n', '2\n', '3\n']}
        file_diff_dict = {}
        diff = Diff(file_dict[self.fa], rename=self.fa + '.renamed')
        diff.change_line(3, '3\n', '3_changed\n')
        ApplyPatchAction().apply(Result('origin', 'msg', diffs={self.fa: diff}), file_dict, file_diff_dict)
        expected_file_dict = {self.fa: ['1\n', '3_changed\n']}
        section = Section('')
        section.append(Setting('editor', 'vim'))
        uut = OpenEditorAction()
        subprocess.call = self.fake_edit
        diff_dict = uut.apply_from_section(Result.from_values('origin', 'msg', self.fa), file_dict, file_diff_dict, section)
        for filename in diff_dict:
            file_dict[filename] = file_diff_dict[filename].modified
        self.assertEqual(file_dict, expected_file_dict)
        open(self.fa, 'w').close()

    def test_is_applicable(self):
        if False:
            while True:
                i = 10
        result1 = Result('', '')
        result2 = Result.from_values('', '', '')
        result3 = Result.from_values('', '', 'file')
        invalid_result = ''
        self.assertEqual(OpenEditorAction.is_applicable(result1, None, {}), 'The result is not associated with any source code.')
        self.assertTrue(OpenEditorAction.is_applicable(result2, None, {}))
        self.assertEqual(OpenEditorAction.is_applicable(result3, None, {}), "The result is associated with source code that doesn't seem to exist.")
        with self.assertRaises(TypeError):
            OpenEditorAction.is_applicable(invalid_result, None, {})

    def test_environ_editor(self):
        if False:
            return 10
        file_dict = {self.fb: ['1\n', '2\n', '3\n']}
        diff_dict = {}
        subprocess.call = self.fake_edit
        with open(self.fb, 'w') as handle:
            handle.writelines(file_dict[self.fb])
        result = Result.from_values('origin', 'msg', self.fb)
        if 'EDITOR' in os.environ:
            del os.environ['EDITOR']
            reload(coalib.results.result_actions.OpenEditorAction)
        from coalib.results.result_actions.OpenEditorAction import OpenEditorAction
        action = OpenEditorAction()
        with self.assertRaises(TypeError):
            action.apply(result, file_dict, diff_dict)
        os.environ['EDITOR'] = 'vim'
        reload(coalib.results.result_actions.OpenEditorAction)
        from coalib.results.result_actions.OpenEditorAction import OpenEditorAction
        action = OpenEditorAction()
        action.apply(result, file_dict, diff_dict)

    def test_open_files_at_position_unknown_editor(self):
        if False:
            print('Hello World!')
        uut = OpenEditorAction()
        result_mock = Result.from_values('test', '', self.fa, line=12, column=8)
        with unittest.mock.patch('subprocess.call') as call:
            uut.apply(result_mock, {self.fa: ''}, {}, editor='unknown_editor')
            call.assert_called_with(['unknown_editor', self.fa])

    def test_open_files_at_position_subl(self):
        if False:
            return 10
        uut = OpenEditorAction()
        result_mock = Result.from_values('test', '', self.fa, line=12, column=8)
        with unittest.mock.patch('subprocess.call') as call:
            uut.apply(result_mock, {self.fa: ''}, {}, editor='subl')
            call.assert_called_with(['subl', '--wait', f'{self.fa}:12:8'], stdout=subprocess.PIPE)

    def test_open_files_at_position_vim(self):
        if False:
            for i in range(10):
                print('nop')
        uut = OpenEditorAction()
        result_mock = Result.from_values('test', '', self.fa, line=12, column=8)
        with unittest.mock.patch('subprocess.call') as call:
            uut.apply(result_mock, {self.fa: ''}, {}, editor='vim')
            call.assert_called_with(['vim', self.fa, '+12'])

    def test_open_files_at_position_no_position(self):
        if False:
            for i in range(10):
                print('nop')
        uut = OpenEditorAction()
        result_mock = Result.from_values('test', '', self.fa, line=None, column=None)
        with unittest.mock.patch('subprocess.call') as call:
            uut.apply(result_mock, {self.fa: ''}, {}, editor='subl')
            call.assert_called_with(['subl', '--wait', f'{self.fa}:1:1'], stdout=subprocess.PIPE)

    def test_unknown_editor_warning(self):
        if False:
            return 10
        logger = logging.getLogger()
        uut = OpenEditorAction()
        result_mock = Result.from_values('test', '', self.fa, line=None, column=None)
        with unittest.mock.patch('subprocess.call'):
            with self.assertLogs(logger, 'WARNING') as log:
                uut.apply(result_mock, {self.fa: ''}, {}, editor='gouda-edit')
                self.assertEqual(1, len(log.output))
                self.assertIn('The editor "gouda-edit" is unknown to coala.', log.output[0])

def test_build_editor_call_args_spaced_filename():
    if False:
        return 10
    uut = OpenEditorAction()
    editor_info = {'args': '--bar --baz', 'file_arg_template': '{filename} +{line}'}
    filenames = {'foo and bar.py': {'filename': 'foo and bar.py', 'line': 10, 'column': 6}}
    assert uut.build_editor_call_args('foo', editor_info, filenames) == ['foo', '--bar', '--baz', 'foo and bar.py', '+10']

def test_build_editor_call_args_multiple_filename():
    if False:
        print('Hello World!')
    uut = OpenEditorAction()
    editor_info = {'file_arg_template': '{filename}:{line}:{column}'}
    filenames = {'foo and bar.py': {'filename': 'foo and bar.py', 'line': 10, 'column': 6}, 'bang bong.py': {'filename': 'bang bong.py', 'line': 14, 'column': 8}}
    assert set(uut.build_editor_call_args('foo', editor_info, filenames)) == set(['foo', 'foo and bar.py:10:6', 'bang bong.py:14:8'])