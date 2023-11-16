import os
import sys
import logging
from coalib.collecting.Collectors import collect_all_bears_from_sections, filter_section_bears_by_languages
from coalib.bearlib.languages.Language import Language, UnknownLanguageError
from coalib.misc import Constants
from coalib.output.ConfWriter import ConfWriter
from coalib.output.printers.LOG_LEVEL import LOG_LEVEL
from coalib.parsing.CliParsing import parse_cli, check_conflicts
from coalib.parsing.ConfParser import ConfParser
from coalib.parsing.DefaultArgParser import PathArg
from coalib.settings.Section import Section, extract_aspects_from_section
from coalib.settings.SectionFilling import fill_settings
from coalib.settings.Setting import Setting, path
from string import Template
COAFILE_OUTPUT = Template("$type '$file' $found!\nHere's what you can do:\n* add `--save` to generate a config file with your current options\n* add `-I` to suppress any use of config files\n")
COARC_OUTPUT = Template("$type '$file' $found!\nHere's what you can do:\nYou can create a .coarc file in your home directory to set certain user wide settings\n")

def aspectize_sections(sections):
    if False:
        i = 10
        return i + 15
    '\n    Search for aspects related setting in a section, initialize it, and then\n    embed the aspects information as AspectList object into the section itself.\n\n    :param sections:  List of section that potentially contain aspects setting.\n    :return:          The new sections.\n    '
    for (_, section) in sections.items():
        if validate_aspect_config(section):
            section.aspects = extract_aspects_from_section(section)
        else:
            section.aspects = None
    return sections

def validate_aspect_config(section):
    if False:
        return 10
    '\n    Validate if a section contain required setting to run in aspects mode.\n\n    :param section: The section that potentially contain aspect\n                    setting.\n    :return:        The validity of section.\n    '
    aspects = section.get('aspects')
    if not len(aspects):
        return False
    if not section.language:
        logging.warning(f'Setting `language` is not found in section `{section.name}`. Usage of aspect-based setting must include language information.')
        return False
    if len(section.get('bears')):
        logging.warning(f'`aspects` and `bears` setting is detected in section `{section.name}`. aspect-based configuration will takes priority and will overwrite any explicitly listed bears.')
    return True

def _set_section_language(sections):
    if False:
        i = 10
        return i + 15
    '\n    Validate ``language`` setting and inject them to section if valid.\n\n    :param sections: List of sections that potentially contain ``language``.\n    '
    for (section_name, section) in sections.items():
        section_language = section.get('language')
        if not len(section_language):
            continue
        try:
            section.language = Language[section_language]
        except UnknownLanguageError as exc:
            logging.warning(f'Section `{section_name}` contain invalid language setting: {exc}')

def merge_section_dicts(lower, higher):
    if False:
        return 10
    '\n    Merges the section dictionaries. The values of higher will take\n    precedence over the ones of lower. Lower will hold the modified dict in\n    the end.\n\n    :param lower:  A section.\n    :param higher: A section which values will take precedence over the ones\n                   from the other.\n    :return:       The merged dict.\n    '
    for name in higher:
        if name in lower:
            lower[name].update(higher[name], ignore_defaults=True)
        else:
            lower[name] = higher[name]
    return lower

def load_config_file(filename, log_printer=None, silent=False):
    if False:
        i = 10
        return i + 15
    "\n    Loads sections from a config file. Prints an appropriate warning if\n    it doesn't exist and returns a section dict containing an empty\n    default section in that case.\n\n    It assumes that the cli_sections are available.\n\n    :param filename:    The file to load settings from.\n    :param log_printer: The log printer to log the warning/error to (in case).\n    :param silent:      Whether or not to warn the user/exit if the file\n                        doesn't exist.\n    :raises SystemExit: Exits when the given filename is invalid and is not the\n                        default coafile. Only raised when ``silent`` is\n                        ``False``.\n    "
    filename = os.path.abspath(filename)
    try:
        return ConfParser().parse(filename)
    except FileNotFoundError:
        if not silent:
            if os.path.basename(filename) == Constants.local_coafile:
                logging.warning(COAFILE_OUTPUT.substitute(type='Local coafile', file=Constants.local_coafile, found='not found'))
            elif os.path.basename(filename) == '.coarc':
                logging.warning(COARC_OUTPUT.substitute(type='Requested coarc file', file=filename, found='does not exist'))
            else:
                logging.error(COAFILE_OUTPUT.substitute(type='Requested coafile', file=filename, found='does not exist'))
                sys.exit(2)
        return {'default': Section('default')}

def save_sections(sections):
    if False:
        for i in range(10):
            print('nop')
    '\n    Saves the given sections if they are to be saved.\n\n    :param sections: A section dict.\n    '
    default_section = sections['cli']
    try:
        if bool(default_section.get('save', 'false')):
            conf_writer = ConfWriter(str(default_section.get('config', Constants.local_coafile)))
        else:
            return
    except ValueError:
        conf_writer = ConfWriter(str(default_section.get('save', '.coafile')))
    conf_writer.write_sections(sections)
    conf_writer.close()

def warn_nonexistent_targets(targets, sections, log_printer=None):
    if False:
        while True:
            i = 10
    '\n    Prints out a warning on the given log printer for all targets that are\n    not existent within the given sections.\n\n    :param targets:     The targets to check.\n    :param sections:    The sections to search. (Dict.)\n    :param log_printer: The log printer to warn to.\n    '
    for target in targets:
        if target not in sections:
            logging.warning(f"The requested section '{target}' is not existent. Thus it cannot be executed.")
    files_config_absent = warn_config_absent(sections, 'files')
    bears_config_absent = warn_config_absent(sections, ['bears', 'aspects'])
    if files_config_absent or bears_config_absent:
        raise SystemExit(2)

def warn_config_absent(sections, argument, log_printer=None):
    if False:
        return 10
    '\n    Checks if at least 1 of the given arguments is present somewhere in the\n    sections and emits a warning that code analysis can not be run without it.\n\n    :param sections:    A dictionary of sections.\n    :param argument:    An argument OR a list of arguments that at least 1\n                        should present.\n    :param log_printer: A log printer to emit the warning to.\n    :return:            Returns a boolean False if the given argument\n                        is present in the sections, else returns True.\n    '
    if isinstance(argument, str):
        argument = [argument]
    for section in sections.values():
        if any((arg in section for arg in argument)):
            return False
    formatted_args = ' or '.join((f'`--{arg}`' for arg in argument))
    logging.warning(f'coala will not run any analysis. Did you forget to give the {formatted_args} argument?')
    return True

def load_configuration(arg_list, log_printer=None, arg_parser=None, args=None, silent=False):
    if False:
        return 10
    '\n    Parses the CLI args and loads the config file accordingly, taking\n    default_coafile and the users .coarc into account.\n\n    :param arg_list:    The list of CLI arguments.\n    :param log_printer: The LogPrinter object for logging.\n    :param arg_parser:  An ``argparse.ArgumentParser`` instance used for\n                        parsing the CLI arguments.\n    :param args:        Alternative pre-parsed CLI arguments.\n    :param silent:      Whether or not to display warnings, ignored if ``save``\n                        is enabled.\n    :return:            A tuple holding (log_printer: LogPrinter, sections:\n                        dict(str, Section), targets: list(str)). (Types\n                        indicated after colon.)\n    '
    cli_sections = parse_cli(arg_list=arg_list, arg_parser=arg_parser, args=args)
    check_conflicts(cli_sections)
    if bool(cli_sections['cli'].get('find_config', 'False')) and str(cli_sections['cli'].get('config')) == '':
        cli_sections['cli'].add_or_create_setting(Setting('config', PathArg(find_user_config(os.getcwd()))))
    targets = [item.lower() for item in list(cli_sections['cli'].contents.pop('targets', ''))]
    if bool(cli_sections['cli'].get('no_config', 'False')):
        sections = cli_sections
    else:
        base_sections = load_config_file(Constants.system_coafile, silent=silent)
        user_sections = load_config_file(Constants.user_coafile, silent=True)
        default_config = str(base_sections['default'].get('config', '.coafile'))
        user_config = str(user_sections['default'].get('config', default_config))
        config = os.path.abspath(str(cli_sections['cli'].get('config', user_config)))
        try:
            save = bool(cli_sections['cli'].get('save', 'False'))
        except ValueError:
            save = True
        coafile_sections = load_config_file(config, silent=save or silent)
        sections = merge_section_dicts(base_sections, user_sections)
        sections = merge_section_dicts(sections, coafile_sections)
        if 'cli' in sections:
            logging.warning("'cli' is an internally reserved section name. It may have been generated into your coafile while running coala with `--save`. The settings in that section will inherit implicitly to all sections as defaults just like CLI args do. Please change the name of that section in your coafile to avoid any unexpected behavior.")
        sections = merge_section_dicts(sections, cli_sections)
    for (name, section) in list(sections.items()):
        section.set_default_section(sections)
        if name == 'default':
            if section.contents:
                logging.warning("Implicit 'Default' section inheritance is deprecated. It will be removed soon. To silence this warning remove settings in the 'Default' section from your coafile. You can use dots to specify inheritance: the section 'all.python' will inherit all settings from 'all'.")
                sections['default'].update(sections['cli'])
                sections['default'].name = 'cli'
                sections['cli'] = sections['default']
            del sections['default']
    str_log_level = str(sections['cli'].get('log_level', '')).upper()
    logging.getLogger().setLevel(LOG_LEVEL.str_dict.get(str_log_level, LOG_LEVEL.INFO))
    return (sections, targets)

def find_user_config(file_path, max_trials=10):
    if False:
        return 10
    "\n    Uses the filepath to find the most suitable user config file for the file\n    by going down one directory at a time and finding config files there.\n\n    :param file_path:  The path of the file whose user config needs to be found\n    :param max_trials: The maximum number of directories to go down to.\n    :return:           The config file's path, empty string if none was found\n    "
    file_path = os.path.normpath(os.path.abspath(os.path.expanduser(file_path)))
    old_dir = None
    base_dir = file_path if os.path.isdir(file_path) else os.path.dirname(file_path)
    home_dir = os.path.expanduser('~')
    while base_dir != old_dir and old_dir != home_dir and (max_trials != 0):
        config_file = os.path.join(base_dir, '.coafile')
        if os.path.isfile(config_file):
            return config_file
        old_dir = base_dir
        base_dir = os.path.dirname(old_dir)
        max_trials = max_trials - 1
    return ''

def get_config_directory(section):
    if False:
        print('Hello World!')
    '\n    Retrieves the configuration directory for the given section.\n\n    Given an empty section:\n\n    >>> section = Section("name")\n\n    The configuration directory is not defined and will therefore fallback to\n    the current directory:\n\n    >>> get_config_directory(section) == os.path.abspath(".")\n    True\n\n    If the ``files`` setting is given with an originating coafile, the directory\n    of the coafile will be assumed the configuration directory:\n\n    >>> section.append(Setting("files", "**", origin="/tmp/.coafile"))\n    >>> get_config_directory(section) == os.path.abspath(\'/tmp/\')\n    True\n\n    However if its origin is already a directory this will be preserved:\n\n    >>> files = Setting(\'files\', \'**\', origin=os.path.abspath(\'/tmp/dir/\'))\n    >>> section.append(files)\n    >>> os.makedirs(section[\'files\'].origin, exist_ok=True)\n    >>> get_config_directory(section) == section[\'files\'].origin\n    True\n\n    The user can manually set a project directory with the ``project_dir``\n    setting:\n\n    >>> section.append(Setting(\'project_dir\', os.path.abspath(\'/tmp\'), \'/\'))\n    >>> get_config_directory(section) == os.path.abspath(\'/tmp\')\n    True\n\n    If no section is given, the current directory is returned:\n\n    >>> get_config_directory(None) == os.path.abspath(".")\n    True\n\n    To summarize, the config directory will be chosen by the following\n    priorities if possible in that order:\n\n    - the ``project_dir`` setting\n    - the origin of the ``files`` setting, if it\'s a directory\n    - the directory of the origin of the ``files`` setting\n    - the current directory\n\n    :param section: The section to inspect.\n    :return: The directory where the project is lying.\n    '
    if section is None:
        return os.getcwd()
    if 'project_dir' in section:
        return path(section.get('project_dir'))
    config = os.path.abspath(section.get('files', '').origin)
    return config if os.path.isdir(config) else os.path.dirname(config)

def get_all_bears(log_printer=None, arg_parser=None, silent=True, bear_globs=('**',)):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param log_printer: The log_printer to handle logging.\n    :param arg_parser:  An ``ArgParser`` object.\n    :param silent:      Whether or not to display warnings.\n    :param bear_globs:  List of glob patterns.\n    :return:            Tuple containing dictionaries of unsorted local\n                        and global bears.\n    '
    (sections, _) = load_configuration(arg_list=None, arg_parser=arg_parser, silent=silent)
    (local_bears, global_bears) = collect_all_bears_from_sections(sections, bear_globs=bear_globs)
    return (local_bears, global_bears)

def get_filtered_bears(languages, log_printer=None, arg_parser=None, silent=True):
    if False:
        print('Hello World!')
    '\n    :param languages:   List of languages.\n    :param log_printer: The log_printer to handle logging.\n    :param arg_parser:  An ``ArgParser`` object.\n    :param silent:      Whether or not to display warnings.\n    :return:            Tuple containing dictionaries of unsorted local\n                        and global bears.\n    '
    (local_bears, global_bears) = get_all_bears(arg_parser=arg_parser, silent=silent)
    if languages:
        local_bears = filter_section_bears_by_languages(local_bears, languages)
        global_bears = filter_section_bears_by_languages(global_bears, languages)
    return (local_bears, global_bears)

def gather_configuration(acquire_settings, log_printer=None, arg_list=None, arg_parser=None, args=None):
    if False:
        print('Hello World!')
    '\n    Loads all configuration files, retrieves bears and all needed\n    settings, saves back if needed and warns about non-existent targets.\n\n    This function:\n\n    -  Reads and merges all settings in sections from\n\n       -  Default config\n       -  User config\n       -  Configuration file\n       -  CLI\n\n    -  Collects all the bears\n    -  Fills up all needed settings\n    -  Writes back the new sections to the configuration file if needed\n    -  Gives all information back to caller\n\n    :param acquire_settings: The method to use for requesting settings. It will\n                             get a parameter which is a dictionary with the\n                             settings name as key and a list containing a\n                             description in [0] and the names of the bears\n                             who need this setting in all following indexes.\n    :param log_printer:      The log printer to use for logging. The log level\n                             will be adjusted to the one given by the section.\n    :param arg_list:         CLI args to use\n    :param arg_parser:       Instance of ArgParser that is used to parse\n                             none-setting arguments.\n    :param args:             Alternative pre-parsed CLI arguments.\n    :return:                 A tuple with the following contents:\n\n                             -  A dictionary with the sections\n                             -  Dictionary of list of local bears for each\n                                section\n                             -  Dictionary of list of global bears for each\n                                section\n                             -  The targets list\n    '
    if args is None:
        arg_list = sys.argv[1:] if arg_list is None else arg_list
    (sections, targets) = load_configuration(arg_list, arg_parser=arg_parser, args=args)
    _set_section_language(sections)
    aspectize_sections(sections)
    (local_bears, global_bears) = fill_settings(sections, acquire_settings, targets=targets)
    save_sections(sections)
    warn_nonexistent_targets(targets, sections)
    return (sections, local_bears, global_bears, targets)