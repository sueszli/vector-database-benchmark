"""
Generates documentation of behave's

  * command-line options
  * configuration-file parameters

REQUIRES: Python >= 2.6
"""
from __future__ import absolute_import, print_function
import re
import sys
import conf
import textwrap
from behave import configuration
from behave.__main__ import TAG_EXPRESSIONS_HELP
positive_number = configuration.positive_number
cmdline = []
config = []
indent = '    '
cmdline_option_schema = '.. option:: {cmdline_option}\n\n    {text}\n'
config_param_schema = '.. index::\n    single: configuration param; {param}\n\n.. describe:: {param} : {type}\n\n    {text}\n'

def is_no_option(fixed_options):
    if False:
        return 10
    return any([opt.startswith('--no') for opt in fixed_options])
for (fixed, keywords) in configuration.OPTIONS:
    skip = False
    config_file_param = True
    if is_no_option(fixed):
        config_file_param = False
    if 'dest' in keywords:
        dest = keywords['dest']
    else:
        for opt in fixed:
            if opt.startswith('--no'):
                skip = True
            if opt.startswith('--'):
                dest = opt[2:].replace('-', '_')
                break
            else:
                assert len(opt) == 2
                dest = opt[1:]
    type_name_default = 'text'
    type_name_map = {'color': 'Colored (Enum)', 'tag_expression_protocol': 'TagExpressionProtocol (Enum)'}
    type_name = 'string'
    action = keywords.get('action', 'store')
    data_type = keywords.get('type', None)
    default_value = keywords.get('default', None)
    if action in ('store', 'store_const'):
        type_name = 'text'
        if data_type is positive_number:
            type_name = 'positive_number'
        elif data_type is int:
            type_name = 'number'
        else:
            type_name = type_name_map.get(dest, type_name_default)
    elif action in ('store_true', 'store_false'):
        type_name = 'bool'
        default_value = False
        if action == 'store_true':
            default_value = True
    elif action == 'append':
        type_name = 'sequence<text>'
    else:
        raise ValueError('unknown action %s' % action)
    text = re.sub('\\s+', ' ', keywords['help']).strip()
    text = text.replace('%%', '%')
    if default_value and '%(default)s' in text:
        text = text.replace('%(default)s', str(default_value))
    text = textwrap.fill(text, 70, initial_indent='', subsequent_indent=indent)
    if fixed:
        cmdline_option = ', '.join(fixed)
        cmdline.append(cmdline_option_schema.format(cmdline_option=cmdline_option, text=text))
    if skip or dest in 'tags_help lang_list lang_help version'.split():
        continue
    if not config_file_param or action == 'store_false':
        continue
    text = re.sub('\\s+', ' ', keywords.get('config_help', keywords['help'])).strip()
    text = text.replace('%%', '%')
    if default_value and '%(default)s' in text:
        text = text.replace('%(default)s', str(default_value))
    text = textwrap.fill(text, 70, initial_indent='', subsequent_indent=indent)
    config.append(config_param_schema.format(param=dest, type=type_name, text=text))
print('Writing behave.rst ...')
with open('behave.rst-template') as f:
    template = f.read()
values = dict(cmdline='\n'.join(cmdline), tag_expression=TAG_EXPRESSIONS_HELP, config='\n'.join(config))
with open('behave.rst', 'w') as f:
    f.write(template.format(**values))