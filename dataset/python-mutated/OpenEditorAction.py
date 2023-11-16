import logging
import shlex
import subprocess
from os.path import exists
from os import environ
from coalib.results.Diff import Diff
from coalib.results.Result import Result
from coalib.results.result_actions.ResultAction import ResultAction
from coala_utils.decorators import enforce_signature
from coala_utils.FileUtils import detect_encoding
'\nData about all text editors coala knows about. New editors\ncan just be added here.\nFor each editor the following info is stored:\n{\n    <name/comand>: {\n        "file_arg_template":\n            A string used to generate arguments to open a file.\n            Must at least have the placeholder \'filename\'\n            and can optionally use \'line\' and \'column\'\n            to open the file at the correct position.\n            Some editors don\'t support opening files at\n            a certain position if multiple files are\n            to be opened, but we try to do so anyway.\n        "args":\n            General arguments added to the call, e.g. to\n            force opening of a new window.\n        "gui":\n            Boolean. True if this is a gui editor.\n            Optional, defaults to False.\n    }\n}\n'
KNOWN_EDITORS = {'vim': {'file_arg_template': '{filename} +{line}', 'gui': False}, 'nvim': {'file_arg_template': '{filename} +{line}', 'gui': False}, 'nano': {'file_arg_template': '+{line},{column} {filename} ', 'gui': False}, 'emacs': {'file_arg_template': '+{line}:{column} {filename}', 'gui': False}, 'emacsclient': {'file_arg_template': '+{line}:{column} {filename}', 'gui': False}, 'atom': {'file_arg_template': '{filename}:{line}:{column}', 'args': '--wait', 'gui': True}, 'code': {'file_arg_template': '{filename}:{line}:{column} --goto', 'args': '--new-window --reuse-window --wait', 'gui': True}, 'geany': {'file_arg_template': '{filename} -l {line} --column {column}', 'args': '-s -i', 'gui': True}, 'gedit': {'file_arg_template': '{filename} +{line}', 'args': '-s', 'gui': True}, 'gvim': {'file_arg_template': '{filename} +{line}', 'gui': True}, 'kate': {'file_arg_template': '{filename} -l {line} -c {column}', 'args': '--new', 'gui': True}, 'notepadqq': {'file_arg_template': '{filename}', 'gui': True}, 'subl': {'file_arg_template': '{filename}:{line}:{column}', 'args': '--wait', 'gui': True}, 'xed': {'file_arg_template': '{filename} +{line}', 'args': '--new-window', 'gui': True}}

class OpenEditorAction(ResultAction):
    SUCCESS_MESSAGE = 'Changes saved successfully.'

    @staticmethod
    @enforce_signature
    def is_applicable(result: Result, original_file_dict, file_diff_dict, applied_actions=()):
        if False:
            while True:
                i = 10
        '\n        For being applicable, the result has to point to a number of files\n        that have to exist i.e. have not been previously deleted.\n        '
        if not len(result.affected_code) > 0:
            return 'The result is not associated with any source code.'
        filenames = set((src.renamed_file(file_diff_dict) for src in result.affected_code))
        if not all((exists(filename) for filename in filenames)):
            return "The result is associated with source code that doesn't seem to exist."
        return True

    def build_editor_call_args(self, editor, editor_info, filenames):
        if False:
            return 10
        '\n        Create argument list which will then be used to open an editor for\n        the given files at the correct positions, if applicable.\n\n        :param editor:\n            The editor to open the file with.\n        :param editor_info:\n            A dict containing the keys ``args`` and ``file_arg_template``,\n            providing additional call arguments and a template to open\n            files at a position for this editor.\n        :param filenames:\n            A dict holding one entry for each file to be opened.\n            Keys must be ``filename``, ``line`` and ``column``.\n        '
        call_args = [editor]
        if 'args' in editor_info:
            call_args += shlex.split(editor_info['args'])
        for file_info in filenames.values():
            file_arg = editor_info['file_arg_template'].format(filename=shlex.quote(file_info['filename']), line=file_info['line'], column=file_info['column'])
            call_args += shlex.split(file_arg)
        return call_args

    def apply(self, result, original_file_dict, file_diff_dict, editor: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        (O)pen file\n\n        :param editor: The editor to open the file with.\n        '
        try:
            editor_info = KNOWN_EDITORS[editor.strip()]
        except KeyError:
            formatted_supported = ', '.join(sorted(KNOWN_EDITORS.keys()))
            logging.warning(f'''The editor "{editor}" is unknown to coala. Files won't be opened at the correct positions and other quirks might occur. Consider opening an issue at https://github.com/coala/coala/issues so we can add support for this editor. Supported editors are: {formatted_supported}''')
            editor_info = {'file_arg_template': '{filename}', 'gui': False}
        filenames = {src.file: {'filename': src.renamed_file(file_diff_dict), 'line': src.start.line or 1, 'column': src.start.column or 1} for src in result.affected_code}
        call_args = self.build_editor_call_args(editor, editor_info, filenames)
        if editor_info.get('gui', True):
            subprocess.call(call_args, stdout=subprocess.PIPE)
        else:
            subprocess.call(call_args)
        for (original_name, file_info) in filenames.items():
            filename = file_info['filename']
            with open(filename, encoding=detect_encoding(filename)) as file:
                file_diff_dict[original_name] = Diff.from_string_arrays(original_file_dict[original_name], file.readlines(), rename=False if original_name == filename else filename)
        return file_diff_dict
    if 'EDITOR' in environ:
        apply.__defaults__ = (environ['EDITOR'],)