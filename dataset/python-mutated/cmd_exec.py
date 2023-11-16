import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List
from flask import Response, request
from werkzeug.utils import secure_filename
from archivy import click_web
from .input_fields import FieldId
logger = None

def exec(command_path):
    if False:
        print('Hello World!')
    '\n    Execute the command and stream the output from it as response\n    :param command_path:\n    '
    command_path = 'cli/' + command_path
    global logger
    logger = click_web.logger
    omitted = ['shell', 'run', 'routes', 'create-admin']
    (root_command, *commands) = command_path.split('/')
    cmd = ['archivy']
    req_to_args = RequestToCommandArgs()
    cmd.extend(req_to_args.command_args(0))
    for (i, command) in enumerate(commands):
        if command in omitted:
            return Response(status=400)
        cmd.append(command)
        cmd.extend(req_to_args.command_args(i + 1))

    def _generate_output():
        if False:
            print('Hello World!')
        yield _create_cmd_header(commands)
        try:
            yield from _run_script_and_generate_stream(req_to_args, cmd)
        except Exception as e:
            yield f'\nERROR: Got exception when reading output from script: {type(e)}\n'
            yield traceback.format_exc()
            raise
    return Response(_generate_output(), mimetype='text/plain')

def _run_script_and_generate_stream(req_to_args: 'RequestToCommandArgs', cmd: List[str]):
    if False:
        while True:
            i = 10
    '\n    Execute the command the via Popen and yield output\n    '
    logger.info('Executing archivy command')
    if not os.environ.get('PYTHONIOENCODING'):
        os.environ['PYTHONIOENCODING'] = 'UTF-8'
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logger.info('script running Pid: %d', process.pid)
    encoding = sys.getdefaultencoding()
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            yield line.decode(encoding)
    process.wait()
    logger.info('script finished Pid: %d', process.pid)
    for fi in req_to_args.field_infos:
        fi.after_script_executed()

def _create_cmd_header(commands: List[str]):
    if False:
        print('Hello World!')
    '\n    Generate a command header.\n    Note:\n        here we always allow to generate HTML as long as we have it between CLICK-WEB comments.\n        This way the JS frontend can insert it in the correct place in the DOM.\n    '

    def generate():
        if False:
            i = 10
            return i + 15
        yield '<!-- CLICK_WEB START HEADER -->'
        yield '<div class="command-line">Executing: {}</div>'.format('/'.join(commands))
        yield '<!-- CLICK_WEB END HEADER -->'
    html_str = '\n'.join(generate())
    return html_str

def _create_result_footer(req_to_args: 'RequestToCommandArgs'):
    if False:
        return 10
    '\n    Generate a footer.\n    Note:\n        here we always allow to generate HTML as long as we have it between CLICK-WEB comments.\n        This way the JS frontend can insert it in the correct place in the DOM.\n    '
    to_download = [fi for fi in req_to_args.field_infos if fi.generate_download_link and fi.link_name]
    lines = []
    lines.append('<!-- CLICK_WEB START FOOTER -->')
    if to_download:
        lines.append('<b>Result files:</b><br>')
        for fi in to_download:
            lines.append('<ul> ')
            lines.append(f'<li>{_get_download_link(fi)}<br>')
            lines.append('</ul>')
    else:
        lines.append('<b>DONE</b>')
    lines.append('<!-- CLICK_WEB END FOOTER -->')
    html_str = '\n'.join(lines)
    yield html_str

def _get_download_link(field_info):
    if False:
        i = 10
        return i + 15
    'Hack as url_for need request context'
    rel_file_path = Path(field_info.file_path).relative_to(click_web.OUTPUT_FOLDER)
    uri = f'/static/results/{rel_file_path.as_posix()}'
    return f'<a href="{uri}">{field_info.link_name}</a>'

class RequestToCommandArgs:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        keys = [key for key in list(request.form.keys()) + list(request.files.keys())]
        field_infos = [FieldInfo.factory(key) for key in keys if key != 'csrf_token']
        self.field_infos = list(sorted(field_infos))

    def command_args(self, command_index) -> List[str]:
        if False:
            return 10
        '\n        Convert the post request into a list of command line arguments\n\n        :param command_index: (int) the index for the command to get arguments for.\n        :return: list of command line arguments for command at that cmd_index\n        '
        args = []
        commands_field_infos = [fi for fi in self.field_infos if fi.param.command_index == command_index]
        commands_field_infos = sorted(commands_field_infos)
        for fi in commands_field_infos:
            fi.before_script_execute()
            if fi.cmd_opt.startswith('--'):
                args.extend(self._process_option(fi))
            elif isinstance(fi, FieldFileInfo):
                args.append(fi.file_path)
            else:
                arg_values = request.form.getlist(fi.key)
                has_values = bool(''.join(arg_values))
                if has_values:
                    if fi.param.nargs == -1:
                        for value in arg_values:
                            values = value.splitlines()
                            logger.info(f'variadic arguments, split into: "{values}"')
                            args.extend(values)
                    else:
                        logger.info(f'arg_value: "{arg_values}"')
                        args.extend(arg_values)
        return args

    def _process_option(self, field_info):
        if False:
            return 10
        vals = request.form.getlist(field_info.key)
        if field_info.is_file:
            if field_info.link_name:
                yield field_info.cmd_opt
                yield field_info.file_path
        elif field_info.param.param_type == 'flag':
            if len(vals) == 1:
                off_flag = vals[0]
                flag_on_cmd_line = off_flag
            else:
                on_flag = vals[1]
                flag_on_cmd_line = on_flag
            yield flag_on_cmd_line
        elif ''.join(vals):
            yield field_info.cmd_opt
            for val in vals:
                if val:
                    yield val
        else:
            pass

class FieldInfo:
    """
    Extract information from the encoded form input field name
    the parts:
        [command_index].[opt_or_arg_index].[click_type].[html_input_type].[opt_or_arg_name]
    e.g.
        "0.0.option.text.text.--an-option"
        "0.1.argument.file[rb].text.an-argument"
    """

    @staticmethod
    def factory(key):
        if False:
            i = 10
            return i + 15
        field_id = FieldId.from_string(key)
        is_file = field_id.click_type.startswith('file')
        is_path = field_id.click_type.startswith('path')
        is_uploaded = key in request.files
        if is_file:
            if is_uploaded:
                field_info = FieldFileInfo(field_id)
            else:
                field_info = FieldOutFileInfo(field_id)
        elif is_path:
            if is_uploaded:
                field_info = FieldPathInfo(field_id)
            else:
                field_info = FieldPathOutInfo(field_id)
        else:
            field_info = FieldInfo(field_id)
        return field_info

    def __init__(self, param: FieldId):
        if False:
            return 10
        self.param = param
        self.key = param.key
        'Type of option (file, text)'
        self.is_file = self.param.click_type.startswith('file')
        'The actual command line option (--debug)'
        self.cmd_opt = param.name
        self.generate_download_link = False

    def before_script_execute(self):
        if False:
            print('Hello World!')
        pass

    def after_script_executed(self):
        if False:
            while True:
                i = 10
        pass

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.param)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        'Make class sortable'
        return (self.param.command_index, self.param.param_index) < (other.param.command_index, other.param.param_index)

    def __eq__(self, other):
        if False:
            return 10
        return self.key == other.key

class FieldFileInfo(FieldInfo):
    """
    Use for processing input fields of file type.
    Saves the posted data to a temp file.
    """
    'temp dir is on class in order to be uniqe for each request'
    _temp_dir = None

    def __init__(self, fimeta):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(fimeta)
        self.mode = self.param.click_type.split('[')[1][:-1]
        self.generate_download_link = True if 'w' in self.mode else False
        self.link_name = f'{self.cmd_opt}.out'
        logger.info(f'File mode for {self.key} is {self.mode}')

    def before_script_execute(self):
        if False:
            for i in range(10):
                print('nop')
        self.save()

    @classmethod
    def temp_dir(cls):
        if False:
            i = 10
            return i + 15
        if not cls._temp_dir:
            cls._temp_dir = tempfile.mkdtemp(dir=click_web.OUTPUT_FOLDER)
        logger.info(f'Temp dir: {cls._temp_dir}')
        return cls._temp_dir

    def save(self):
        if False:
            i = 10
            return i + 15
        logger.info('Saving...')
        logger.info('field value is a file! %s', self.key)
        file = request.files[self.key]
        if file.filename == '':
            raise ValueError('No selected file')
        elif file and file.filename:
            filename = secure_filename(file.filename)
            (name, suffix) = os.path.splitext(filename)
            (fd, filename) = tempfile.mkstemp(dir=self.temp_dir(), prefix=name, suffix=suffix)
            self.file_path = filename
            logger.info(f'Saving {self.key} to {filename}')
            file.save(filename)

    def __str__(self):
        if False:
            while True:
                i = 10
        res = [super().__str__()]
        res.append(f'file_path: {self.file_path}')
        return ', '.join(res)

class FieldOutFileInfo(FieldFileInfo):
    """
    Used when file option is just for output and form posted it as hidden or text field.
    Just create a empty temp file to give it's path to command.
    """

    def __init__(self, fimeta):
        if False:
            i = 10
            return i + 15
        super().__init__(fimeta)
        if self.param.form_type == 'text':
            self.link_name = request.form[self.key]
            self.file_suffix = request.form[self.key]
        else:
            self.file_suffix = '.out'

    def save(self):
        if False:
            print('Hello World!')
        name = secure_filename(self.key)
        filename = tempfile.mkstemp(dir=self.temp_dir(), prefix=name, suffix=self.file_suffix)
        logger.info(f'Creating empty file for {self.key} as {filename}')
        self.file_path = filename

class FieldPathInfo(FieldFileInfo):
    """
    Use for processing input fields of path type.
    Extracts the posted data to a temp folder.
    When script finished zip that folder and provide download link to zip file.
    """

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        super().save()
        zip_extract_dir = tempfile.mkdtemp(dir=self.temp_dir())
        logger.info(f'Extracting: {self.file_path} to {zip_extract_dir}')
        shutil.unpack_archive(self.file_path, zip_extract_dir, 'zip')
        self.file_path = zip_extract_dir

    def after_script_executed(self):
        if False:
            while True:
                i = 10
        super().after_script_executed()
        (fd, filename) = tempfile.mkstemp(dir=self.temp_dir(), prefix=self.key)
        folder_path = self.file_path
        self.file_path = filename
        logger.info(f'Zipping {self.key} to {filename}')
        self.file_path = shutil.make_archive(self.file_path, 'zip', folder_path)
        logger.info(f'Zip file created {self.file_path}')
        self.generate_download_link = True

class FieldPathOutInfo(FieldOutFileInfo):
    """
    Use for processing output fields of path type.
    Create a folder and use as path to script.
    When script finished zip that folder and provide download link to zip file.
    """

    def save(self):
        if False:
            while True:
                i = 10
        super().save()
        self.file_path = tempfile.mkdtemp(dir=self.temp_dir())

    def after_script_executed(self):
        if False:
            while True:
                i = 10
        super().after_script_executed()
        (fd, filename) = tempfile.mkstemp(dir=self.temp_dir(), prefix=self.key)
        folder_path = self.file_path
        self.file_path = filename
        logger.info(f'Zipping {self.key} to {filename}')
        self.file_path = shutil.make_archive(self.file_path, 'zip', folder_path)
        logger.info(f'Zip file created {self.file_path}')
        self.generate_download_link = True