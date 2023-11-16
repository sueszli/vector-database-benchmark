"""
SaltStack Extend
~~~~~~~~~~~~~~~~

A templating tool for extending SaltStack.

Takes a template directory and merges it into a SaltStack source code
directory. This tool uses Jinja2 for templating.

This tool is accessed using `salt-extend`

    :codeauthor: Anthony Shaw <anthonyshaw@apache.org>
"""
import logging
import os
import shutil
import sys
import tempfile
from datetime import date
from jinja2 import Template
import salt.utils.files
import salt.version
from salt.serializers.yaml import deserialize
from salt.utils.odict import OrderedDict
log = logging.getLogger(__name__)
try:
    import click
    HAS_CLICK = True
except ImportError as ie:
    HAS_CLICK = False
TEMPLATE_FILE_NAME = 'template.yml'

def _get_template(path, option_key):
    if False:
        i = 10
        return i + 15
    '\n    Get the contents of a template file and provide it as a module type\n\n    :param path: path to the template.yml file\n    :type  path: ``str``\n\n    :param option_key: The unique key of this template\n    :type  option_key: ``str``\n\n    :returns: Details about the template\n    :rtype: ``tuple``\n    '
    with salt.utils.files.fopen(path, 'r') as template_f:
        template = deserialize(template_f)
        info = (option_key, template.get('description', ''), template)
    return info

def _fetch_templates(src):
    if False:
        return 10
    "\n    Fetch all of the templates in the src directory\n\n    :param src: The source path\n    :type  src: ``str``\n\n    :rtype: ``list`` of ``tuple``\n    :returns: ``list`` of ('key', 'description')\n    "
    templates = []
    log.debug('Listing contents of %s', src)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        if os.path.isdir(s):
            template_path = os.path.join(s, TEMPLATE_FILE_NAME)
            if os.path.isfile(template_path):
                templates.append(_get_template(template_path, item))
            else:
                log.debug('Directory does not contain %s %s', template_path, TEMPLATE_FILE_NAME)
    return templates

def _mergetree(src, dst):
    if False:
        i = 10
        return i + 15
    '\n    Akin to shutils.copytree but over existing directories, does a recursive merge copy.\n\n    :param src: The source path\n    :type  src: ``str``\n\n    :param dst: The destination path\n    :type  dst: ``str``\n    '
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            log.info('Copying folder %s to %s', s, d)
            if os.path.exists(d):
                _mergetree(s, d)
            else:
                shutil.copytree(s, d)
        else:
            log.info('Copying file %s to %s', s, d)
            shutil.copy2(s, d)

def _mergetreejinja(src, dst, context):
    if False:
        while True:
            i = 10
    '\n    Merge directory A to directory B, apply Jinja2 templating to both\n    the file/folder names AND to the contents of the files\n\n    :param src: The source path\n    :type  src: ``str``\n\n    :param dst: The destination path\n    :type  dst: ``str``\n\n    :param context: The dictionary to inject into the Jinja template as context\n    :type  context: ``dict``\n    '
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            log.info('Copying folder %s to %s', s, d)
            if os.path.exists(d):
                _mergetreejinja(s, d, context)
            else:
                os.mkdir(d)
                _mergetreejinja(s, d, context)
        elif item != TEMPLATE_FILE_NAME:
            d = Template(d).render(context)
            log.info('Copying file %s to %s', s, d)
            with salt.utils.files.fopen(s, 'r') as source_file:
                src_contents = salt.utils.stringutils.to_unicode(source_file.read())
                dest_contents = Template(src_contents).render(context)
            with salt.utils.files.fopen(d, 'w') as dest_file:
                dest_file.write(salt.utils.stringutils.to_str(dest_contents))

def _prompt_user_variable(var_name, default_value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prompt the user to enter the value of a variable\n\n    :param var_name: The question to ask the user\n    :type  var_name: ``str``\n\n    :param default_value: The default value\n    :type  default_value: ``str``\n\n    :rtype: ``str``\n    :returns: the value from the user\n    '
    return click.prompt(var_name, default=default_value)

def _prompt_choice(var_name, options):
    if False:
        i = 10
        return i + 15
    '\n    Prompt the user to choose between a list of options, index each one by adding an enumerator\n    based on https://github.com/audreyr/cookiecutter/blob/master/cookiecutter/prompt.py#L51\n\n    :param var_name: The question to ask the user\n    :type  var_name: ``str``\n\n    :param options: A list of options\n    :type  options: ``list`` of ``tupple``\n\n    :rtype: ``tuple``\n    :returns: The selected user\n    '
    choice_map = OrderedDict((('{}'.format(i), value) for (i, value) in enumerate(options, 1) if value[0] != 'test'))
    choices = choice_map.keys()
    default = '1'
    choice_lines = ['{} - {} - {}'.format(c[0], c[1][0], c[1][1]) for c in choice_map.items()]
    prompt = '\n'.join(('Select {}:'.format(var_name), '\n'.join(choice_lines), 'Choose from {}'.format(', '.join(choices))))
    user_choice = click.prompt(prompt, type=click.Choice(choices), default=default)
    return choice_map[user_choice]

def apply_template(template_dir, output_dir, context):
    if False:
        i = 10
        return i + 15
    '\n    Apply the template from the template directory to the output\n    using the supplied context dict.\n\n    :param src: The source path\n    :type  src: ``str``\n\n    :param dst: The destination path\n    :type  dst: ``str``\n\n    :param context: The dictionary to inject into the Jinja template as context\n    :type  context: ``dict``\n    '
    _mergetreejinja(template_dir, output_dir, context)

def run(extension=None, name=None, description=None, salt_dir=None, merge=False, temp_dir=None):
    if False:
        return 10
    "\n    A template factory for extending the salt ecosystem\n\n    :param extension: The extension type, e.g. 'module', 'state', if omitted, user will be prompted\n    :type  extension: ``str``\n\n    :param name: Python-friendly name for the module, if omitted, user will be prompted\n    :type  name: ``str``\n\n    :param description: A description of the extension, if omitted, user will be prompted\n    :type  description: ``str``\n\n    :param salt_dir: The targeted Salt source directory\n    :type  salt_dir: ``str``\n\n    :param merge: Merge with salt directory, `False` to keep separate, `True` to merge trees.\n    :type  merge: ``bool``\n\n    :param temp_dir: The directory for generated code, if omitted, system temp will be used\n    :type  temp_dir: ``str``\n    "
    if not HAS_CLICK:
        print('click is not installed, please install using pip')
        sys.exit(1)
    if salt_dir is None:
        salt_dir = '.'
    MODULE_OPTIONS = _fetch_templates(os.path.join(salt_dir, 'templates'))
    if extension is None:
        print('Choose which type of extension you are developing for SaltStack')
        extension_type = 'Extension type'
        chosen_extension = _prompt_choice(extension_type, MODULE_OPTIONS)
    else:
        if extension not in list(zip(*MODULE_OPTIONS))[0]:
            print('Module extension option not valid')
            sys.exit(1)
        chosen_extension = [m for m in MODULE_OPTIONS if m[0] == extension][0]
    extension_type = chosen_extension[0]
    extension_context = chosen_extension[2]
    if name is None:
        print('Enter the short name for the module (e.g. mymodule)')
        name = _prompt_user_variable('Module name', '')
    if description is None:
        description = _prompt_user_variable('Short description of the module', '')
    template_dir = 'templates/{}'.format(extension_type)
    module_name = name
    param_dict = {'version': salt.version.SaltStackVersion.next_release().name, 'module_name': module_name, 'short_description': description, 'release_date': date.today().strftime('%Y-%m-%d'), 'year': date.today().strftime('%Y')}
    additional_context = {}
    for (key, val) in extension_context.get('questions', {}).items():
        default = Template(val.get('default', '')).render(param_dict)
        prompt_var = _prompt_user_variable(val['question'], default)
        additional_context[key] = prompt_var
    context = param_dict.copy()
    context.update(extension_context)
    context.update(additional_context)
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    apply_template(template_dir, temp_dir, context)
    if not merge:
        path = temp_dir
    else:
        _mergetree(temp_dir, salt_dir)
        path = salt_dir
    log.info('New module stored in %s', path)
    return path
if __name__ == '__main__':
    run()