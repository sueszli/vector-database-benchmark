import os
from argparse import ArgumentParser
from collections import OrderedDict
from coalib.parsing.DefaultArgParser import default_arg_parser
from coalib.parsing.LineParser import LineParser
from coalib.settings.Section import Section, append_to_sections
from coalib.bearlib import deprecate_settings

@deprecate_settings(comment_separators='comment_seperators')
def parse_cli(arg_list=None, origin=os.getcwd(), arg_parser=None, args=None, key_value_delimiters=('=', ':'), comment_separators=(), key_delimiters=(',',), section_override_delimiters=('.',), key_value_append_delimiters=('+=',)):
    if False:
        i = 10
        return i + 15
    "\n    Parses the CLI arguments and creates sections out of it.\n\n    :param arg_list:                    The CLI argument list.\n    :param origin:                      Directory used to interpret relative\n                                        paths given as argument.\n    :param arg_parser:                  Instance of ArgParser that is used to\n                                        parse none-setting arguments.\n    :param args:                        Alternative pre-parsed CLI arguments.\n    :param key_value_delimiters:        Delimiters to separate key and value\n                                        in setting arguments where settings are\n                                        being defined.\n    :param comment_separators:          Allowed prefixes for comments.\n    :param key_delimiters:              Delimiter to separate multiple keys of\n                                        a setting argument.\n    :param section_override_delimiters: The delimiter to delimit the section\n                                        from the key name (e.g. the '.' in\n                                        sect.key = value).\n    :param key_value_append_delimiters: Delimiters to separate key and value\n                                        in setting arguments where settings are\n                                        being appended.\n    :return:                            A dictionary holding section names\n                                        as keys and the sections themselves\n                                        as value.\n    "
    assert not (arg_list and args), 'Either call parse_cli() with an arg_list of CLI arguments or with pre-parsed args, but not with both.'
    if args is None:
        arg_parser = default_arg_parser() if arg_parser is None else arg_parser
        args = arg_parser.parse_args(arg_list)
    origin += os.path.sep
    sections = OrderedDict(cli=Section('cli'))
    line_parser = LineParser(key_value_delimiters, comment_separators, key_delimiters, {}, section_override_delimiters, key_value_append_delimiters)
    for (arg_key, arg_value) in sorted(vars(args).items()):
        if arg_key == 'settings' and arg_value is not None:
            parse_custom_settings(sections, arg_value, origin, line_parser)
        else:
            if isinstance(arg_value, list):
                arg_value = ','.join([str(val) for val in arg_value])
            append_to_sections(sections, arg_key, arg_value, origin, section_name='cli', from_cli=True)
    return sections

def parse_custom_settings(sections, custom_settings_list, origin, line_parser):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses the custom settings given to coala via ``-S something=value``.\n\n    :param sections:             The Section dictionary to add to (mutable).\n    :param custom_settings_list: The list of settings strings.\n    :param origin:               The originating directory.\n    :param line_parser:          The LineParser to use.\n    '
    for setting_definition in custom_settings_list:
        (_, key_tuples, value, append, _) = line_parser._parse(setting_definition)
        for key_tuple in key_tuples:
            append_to_sections(sections, key=key_tuple[1], value=value, origin=origin, to_append=append, section_name=key_tuple[0] or 'cli', from_cli=True)

def check_conflicts(sections):
    if False:
        return 10
    '\n    Checks if there are any conflicting arguments passed.\n\n    :param sections:    The ``{section_name: section_object}`` dictionary to\n                        check conflicts for.\n    :return:            True if no conflicts occur.\n    :raises SystemExit: If there are conflicting arguments (exit code: 2)\n    '
    for section in sections.values():
        if section.get('no_config', False) and (section.get('save', False) or section.get('find_config', False) or str(section.get('config', 'input')) != 'input'):
            ArgumentParser().error("'no_config' cannot be set together with 'save', 'find_config' or 'config'.")
        if not section.get('json', False) and (str(section.get('output', '')) or section.get('relpath', False)):
            ArgumentParser().error("'output' or 'relpath' cannot be used without `--json`.")
    return True