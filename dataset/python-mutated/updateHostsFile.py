import argparse
import fnmatch
import ipaddress
import json
import locale
import os
import platform
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from glob import glob
from typing import Optional, Tuple
PY3 = sys.version_info >= (3, 0)
if not PY3:
    raise Exception('We do not support Python 2 anymore.')
try:
    import requests
except ImportError:
    raise ImportError("This project's dependencies have changed. The Requests library (https://docs.python-requests.org/en/latest/) is now required.")
if platform.system() == 'OpenBSD':
    SUDO = ['/usr/bin/doas']
elif platform.system() == 'Windows':
    SUDO = ['powershell', 'Start-Process', 'powershell', '-Verb', 'runAs']
else:
    SUDO = ['/usr/bin/env', 'sudo']
BASEDIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_defaults():
    if False:
        print('Hello World!')
    '\n    Helper method for getting the default settings.\n\n    Returns\n    -------\n    default_settings : dict\n        A dictionary of the default settings when updating host information.\n    '
    return {'numberofrules': 0, 'datapath': path_join_robust(BASEDIR_PATH, 'data'), 'freshen': True, 'replace': False, 'backup': False, 'skipstatichosts': False, 'keepdomaincomments': True, 'extensionspath': path_join_robust(BASEDIR_PATH, 'extensions'), 'extensions': [], 'nounifiedhosts': False, 'compress': False, 'minimise': False, 'outputsubfolder': '', 'hostfilename': 'hosts', 'targetip': '0.0.0.0', 'sourcedatafilename': 'update.json', 'sourcesdata': [], 'readmefilename': 'readme.md', 'readmetemplate': path_join_robust(BASEDIR_PATH, 'readme_template.md'), 'readmedata': {}, 'readmedatafilename': path_join_robust(BASEDIR_PATH, 'readmeData.json'), 'exclusionpattern': '([a-zA-Z\\d-]+\\.){0,}', 'exclusionregexes': [], 'exclusions': [], 'commonexclusions': ['hulu.com'], 'blacklistfile': path_join_robust(BASEDIR_PATH, 'blacklist'), 'whitelistfile': path_join_robust(BASEDIR_PATH, 'whitelist')}

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Creates a unified hosts file from hosts stored in the data subfolders.')
    parser.add_argument('--auto', '-a', dest='auto', default=False, action='store_true', help='Run without prompting.')
    parser.add_argument('--backup', '-b', dest='backup', default=False, action='store_true', help='Backup the hosts files before they are overridden.')
    parser.add_argument('--extensions', '-e', dest='extensions', default=[], nargs='*', help='Host extensions to include in the final hosts file.')
    parser.add_argument('--nounifiedhosts', dest='nounifiedhosts', default=False, action='store_true', help='Do not include the unified hosts file in the final hosts file. Usually used together with `--extensions`.')
    parser.add_argument('--ip', '-i', dest='targetip', default='0.0.0.0', help='Target IP address. Default is 0.0.0.0.')
    parser.add_argument('--keepdomaincomments', '-k', dest='keepdomaincomments', action='store_false', default=True, help='Do not keep domain line comments.')
    parser.add_argument('--noupdate', '-n', dest='noupdate', default=False, action='store_true', help="Don't update from host data sources.")
    parser.add_argument('--skipstatichosts', '-s', dest='skipstatichosts', default=False, action='store_true', help='Skip static localhost entries in the final hosts file.')
    parser.add_argument('--nogendata', '-g', dest='nogendata', default=False, action='store_true', help='Skip generation of readmeData.json')
    parser.add_argument('--output', '-o', dest='outputsubfolder', default='', help='Output subfolder for generated hosts file.')
    parser.add_argument('--replace', '-r', dest='replace', default=False, action='store_true', help='Replace your active hosts file with this new hosts file.')
    parser.add_argument('--flush-dns-cache', '-f', dest='flushdnscache', default=False, action='store_true', help='Attempt to flush DNS cache after replacing the hosts file.')
    parser.add_argument('--compress', '-c', dest='compress', default=False, action='store_true', help='Compress the hosts file ignoring non-necessary lines (empty lines and comments) and putting multiple domains in each line. Improve the performance under Windows.')
    parser.add_argument('--minimise', '-m', dest='minimise', default=False, action='store_true', help='Minimise the hosts file ignoring non-necessary lines (empty lines and comments).')
    parser.add_argument('--whitelist', '-w', dest='whitelistfile', default=path_join_robust(BASEDIR_PATH, 'whitelist'), help='Whitelist file to use while generating hosts files.')
    parser.add_argument('--blacklist', '-x', dest='blacklistfile', default=path_join_robust(BASEDIR_PATH, 'blacklist'), help='Blacklist file to use while generating hosts files.')
    global settings
    options = vars(parser.parse_args())
    options['outputpath'] = path_join_robust(BASEDIR_PATH, options['outputsubfolder'])
    options['freshen'] = not options['noupdate']
    settings = get_defaults()
    settings.update(options)
    data_path = settings['datapath']
    extensions_path = settings['extensionspath']
    settings['sources'] = list_dir_no_hidden(data_path)
    settings['extensionsources'] = list_dir_no_hidden(extensions_path)
    settings['extensions'] = [os.path.basename(item) for item in list_dir_no_hidden(extensions_path)]
    settings['extensions'] = sorted(list(set(options['extensions']).intersection(settings['extensions'])))
    auto = settings['auto']
    exclusion_regexes = settings['exclusionregexes']
    source_data_filename = settings['sourcedatafilename']
    no_unified_hosts = settings['nounifiedhosts']
    update_sources = prompt_for_update(freshen=settings['freshen'], update_auto=auto)
    if update_sources:
        update_all_sources(source_data_filename, settings['hostfilename'])
    gather_exclusions = prompt_for_exclusions(skip_prompt=auto)
    if gather_exclusions:
        common_exclusions = settings['commonexclusions']
        exclusion_pattern = settings['exclusionpattern']
        exclusion_regexes = display_exclusion_options(common_exclusions=common_exclusions, exclusion_pattern=exclusion_pattern, exclusion_regexes=exclusion_regexes)
    extensions = settings['extensions']
    sources_data = update_sources_data(settings['sourcesdata'], datapath=data_path, extensions=extensions, extensionspath=extensions_path, sourcedatafilename=source_data_filename, nounifiedhosts=no_unified_hosts)
    merge_file = create_initial_file(nounifiedhosts=no_unified_hosts)
    remove_old_hosts_file(settings['outputpath'], 'hosts', settings['backup'])
    if settings['compress']:
        final_file = open(path_join_robust(settings['outputpath'], 'hosts'), 'w+b')
        compressed_file = tempfile.NamedTemporaryFile()
        remove_dups_and_excl(merge_file, exclusion_regexes, compressed_file)
        compress_file(compressed_file, settings['targetip'], final_file)
    elif settings['minimise']:
        final_file = open(path_join_robust(settings['outputpath'], 'hosts'), 'w+b')
        minimised_file = tempfile.NamedTemporaryFile()
        remove_dups_and_excl(merge_file, exclusion_regexes, minimised_file)
        minimise_file(minimised_file, settings['targetip'], final_file)
    else:
        final_file = remove_dups_and_excl(merge_file, exclusion_regexes)
    number_of_rules = settings['numberofrules']
    output_subfolder = settings['outputsubfolder']
    skip_static_hosts = settings['skipstatichosts']
    write_opening_header(final_file, extensions=extensions, numberofrules=number_of_rules, outputsubfolder=output_subfolder, skipstatichosts=skip_static_hosts, nounifiedhosts=no_unified_hosts)
    final_file.close()
    if not settings['nogendata']:
        update_readme_data(settings['readmedatafilename'], extensions=extensions, numberofrules=number_of_rules, outputsubfolder=output_subfolder, sourcesdata=sources_data, nounifiedhosts=no_unified_hosts)
    print_success('Success! The hosts file has been saved in folder ' + output_subfolder + '\nIt contains ' + '{:,}'.format(number_of_rules) + ' unique entries.')
    move_file = prompt_for_move(final_file, auto=auto, replace=settings['replace'], skipstatichosts=skip_static_hosts)
    if move_file:
        prompt_for_flush_dns_cache(flush_cache=settings['flushdnscache'], prompt_flush=not auto)

def prompt_for_update(freshen, update_auto):
    if False:
        i = 10
        return i + 15
    '\n    Prompt the user to update all hosts files.\n\n    If requested, the function will update all data sources after it\n    checks that a hosts file does indeed exist.\n\n    Parameters\n    ----------\n    freshen : bool\n        Whether data sources should be updated. This function will return\n        if it is requested that data sources not be updated.\n    update_auto : bool\n        Whether or not to automatically update all data sources.\n\n    Returns\n    -------\n    update_sources : bool\n        Whether or not we should update data sources for exclusion files.\n    '
    hosts_file = path_join_robust(BASEDIR_PATH, 'hosts')
    if not os.path.isfile(hosts_file):
        try:
            open(hosts_file, 'w+').close()
        except (IOError, OSError):
            print_failure("ERROR: No 'hosts' file in the folder. Try creating one manually.")
    if not freshen:
        return
    prompt = 'Do you want to update all data sources?'
    if update_auto or query_yes_no(prompt):
        return True
    elif not update_auto:
        print("OK, we'll stick with what we've got locally.")
    return False

def prompt_for_exclusions(skip_prompt):
    if False:
        return 10
    '\n    Prompt the user to exclude any custom domains from being blocked.\n\n    Parameters\n    ----------\n    skip_prompt : bool\n        Whether or not to skip prompting for custom domains to be excluded.\n        If true, the function returns immediately.\n\n    Returns\n    -------\n    gather_exclusions : bool\n        Whether or not we should proceed to prompt the user to exclude any\n        custom domains beyond those in the whitelist.\n    '
    prompt = 'Do you want to exclude any domains?\nFor example, hulu.com video streaming must be able to access its tracking and ad servers in order to play video.'
    if not skip_prompt:
        if query_yes_no(prompt):
            return True
        else:
            print("OK, we'll only exclude domains in the whitelist.")
    return False

def prompt_for_flush_dns_cache(flush_cache, prompt_flush):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prompt the user to flush the DNS cache.\n\n    Parameters\n    ----------\n    flush_cache : bool\n        Whether to flush the DNS cache without prompting.\n    prompt_flush : bool\n        If `flush_cache` is False, whether we should prompt for flushing the\n        cache. Otherwise, the function returns immediately.\n    '
    if flush_cache:
        flush_dns_cache()
    elif prompt_flush:
        if query_yes_no('Attempt to flush the DNS cache?'):
            flush_dns_cache()

def prompt_for_move(final_file, **move_params):
    if False:
        return 10
    '\n    Prompt the user to move the newly created hosts file to its designated\n    location in the OS.\n\n    Parameters\n    ----------\n    final_file : file\n        The file object that contains the newly created hosts data.\n    move_params : kwargs\n        Dictionary providing additional parameters for moving the hosts file\n        into place. Currently, those fields are:\n\n        1) auto\n        2) replace\n        3) skipstatichosts\n\n    Returns\n    -------\n    move_file : bool\n        Whether or not the final hosts file was moved.\n    '
    skip_static_hosts = move_params['skipstatichosts']
    if move_params['replace'] and (not skip_static_hosts):
        move_file = True
    elif move_params['auto'] or skip_static_hosts:
        move_file = False
    else:
        prompt = 'Do you want to replace your existing hosts file with the newly generated file?'
        move_file = query_yes_no(prompt)
    if move_file:
        move_file = move_hosts_file_into_place(final_file)
    return move_file

def sort_sources(sources):
    if False:
        while True:
            i = 10
    "\n    Sorts the sources.\n    The idea is that all Steven Black's list, file or entries\n    get on top and the rest sorted alphabetically.\n\n    Parameters\n    ----------\n    sources: list\n        The sources to sort.\n    "
    result = sorted(sources.copy(), key=lambda x: x.lower().replace('-', '').replace('_', '').replace(' ', ''))
    steven_black_positions = [x for (x, y) in enumerate(result) if 'stevenblack' in y.lower()]
    for index in steven_black_positions:
        result.insert(0, result.pop(index))
    return result

def display_exclusion_options(common_exclusions, exclusion_pattern, exclusion_regexes):
    if False:
        while True:
            i = 10
    '\n    Display the exclusion options to the user.\n\n    This function checks whether a user wants to exclude particular domains,\n    and if so, excludes them.\n\n    Parameters\n    ----------\n    common_exclusions : list\n        A list of common domains that are excluded from being blocked. One\n        example is Hulu. This setting is set directly in the script and cannot\n        be overwritten by the user.\n    exclusion_pattern : str\n        The exclusion pattern with which to create the domain regex.\n    exclusion_regexes : list\n        The list of regex patterns used to exclude domains.\n\n    Returns\n    -------\n    aug_exclusion_regexes : list\n        The original list of regex patterns potentially with additional\n        patterns from domains that the user chooses to exclude.\n    '
    for exclusion_option in common_exclusions:
        prompt = 'Do you want to exclude the domain ' + exclusion_option + ' ?'
        if query_yes_no(prompt):
            exclusion_regexes = exclude_domain(exclusion_option, exclusion_pattern, exclusion_regexes)
        else:
            continue
    if query_yes_no('Do you want to exclude any other domains?'):
        exclusion_regexes = gather_custom_exclusions(exclusion_pattern, exclusion_regexes)
    return exclusion_regexes

def gather_custom_exclusions(exclusion_pattern, exclusion_regexes):
    if False:
        return 10
    '\n    Gather custom exclusions from the user.\n\n    Parameters\n    ----------\n    exclusion_pattern : str\n        The exclusion pattern with which to create the domain regex.\n    exclusion_regexes : list\n        The list of regex patterns used to exclude domains.\n\n    Returns\n    -------\n    aug_exclusion_regexes : list\n        The original list of regex patterns potentially with additional\n        patterns from domains that the user chooses to exclude.\n    '
    while True:
        domain_prompt = 'Enter the domain you want to exclude (e.g. facebook.com): '
        user_domain = input(domain_prompt)
        if is_valid_user_provided_domain_format(user_domain):
            exclusion_regexes = exclude_domain(user_domain, exclusion_pattern, exclusion_regexes)
        continue_prompt = 'Do you have more domains you want to enter?'
        if not query_yes_no(continue_prompt):
            break
    return exclusion_regexes

def exclude_domain(domain, exclusion_pattern, exclusion_regexes):
    if False:
        i = 10
        return i + 15
    '\n    Exclude a domain from being blocked.\n\n    This creates the domain regex by which to exclude this domain and appends\n    it a list of already-existing exclusion regexes.\n\n    Parameters\n    ----------\n    domain : str\n        The filename or regex pattern to exclude.\n    exclusion_pattern : str\n        The exclusion pattern with which to create the domain regex.\n    exclusion_regexes : list\n        The list of regex patterns used to exclude domains.\n\n    Returns\n    -------\n    aug_exclusion_regexes : list\n        The original list of regex patterns with one additional pattern from\n        the `domain` input.\n    '
    exclusion_regex = re.compile(exclusion_pattern + domain)
    exclusion_regexes.append(exclusion_regex)
    return exclusion_regexes

def matches_exclusions(stripped_rule, exclusion_regexes):
    if False:
        print('Hello World!')
    '\n    Check whether a rule matches an exclusion rule we already provided.\n\n    If this function returns True, that means this rule should be excluded\n    from the final hosts file.\n\n    Parameters\n    ----------\n    stripped_rule : str\n        The rule that we are checking.\n    exclusion_regexes : list\n        The list of regex patterns used to exclude domains.\n\n    Returns\n    -------\n    matches_exclusion : bool\n        Whether or not the rule string matches a provided exclusion.\n    '
    try:
        stripped_domain = stripped_rule.split()[1]
    except IndexError:
        stripped_domain = stripped_rule
    for exclusionRegex in exclusion_regexes:
        if exclusionRegex.search(stripped_domain):
            return True
    return False

def update_sources_data(sources_data, **sources_params):
    if False:
        i = 10
        return i + 15
    '\n    Update the sources data and information for each source.\n\n    Parameters\n    ----------\n    sources_data : list\n        The list of sources data that we are to update.\n    sources_params : kwargs\n        Dictionary providing additional parameters for updating the\n        sources data. Currently, those fields are:\n\n        1) datapath\n        2) extensions\n        3) extensionspath\n        4) sourcedatafilename\n        5) nounifiedhosts\n\n    Returns\n    -------\n    update_sources_data : list\n        The original source data list with new source data appended.\n    '
    source_data_filename = sources_params['sourcedatafilename']
    if not sources_params['nounifiedhosts']:
        for source in sort_sources(recursive_glob(sources_params['datapath'], source_data_filename)):
            update_file = open(source, 'r', encoding='UTF-8')
            try:
                update_data = json.load(update_file)
                sources_data.append(update_data)
            finally:
                update_file.close()
    for source in sources_params['extensions']:
        source_dir = path_join_robust(sources_params['extensionspath'], source)
        for update_file_path in sort_sources(recursive_glob(source_dir, source_data_filename)):
            update_file = open(update_file_path, 'r')
            try:
                update_data = json.load(update_file)
                sources_data.append(update_data)
            finally:
                update_file.close()
    return sources_data

def jsonarray(json_array_string):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transformer, converts a json array string hosts into one host per\n    line, prefixing each line with "127.0.0.1 ".\n\n    Parameters\n    ----------\n    json_array_string : str\n        The json array string in the form\n          \'["example1.com", "example1.com", ...]\'\n    '
    temp_list = json.loads(json_array_string)
    hostlines = '127.0.0.1 ' + '\n127.0.0.1 '.join(temp_list)
    return hostlines

def update_all_sources(source_data_filename, host_filename):
    if False:
        print('Hello World!')
    '\n    Update all host files, regardless of folder depth.\n\n    Parameters\n    ----------\n    source_data_filename : str\n        The name of the filename where information regarding updating\n        sources for a particular URL is stored. This filename is assumed\n        to be the same for all sources.\n    host_filename : str\n        The name of the file in which the updated source information\n        is stored for a particular URL. This filename is assumed to be\n        the same for all sources.\n    '
    transform_methods = {'jsonarray': jsonarray}
    all_sources = sort_sources(recursive_glob('*', source_data_filename))
    for source in all_sources:
        update_file = open(source, 'r', encoding='UTF-8')
        update_data = json.load(update_file)
        update_file.close()
        if update_data.get('pause', False):
            continue
        update_url = update_data['url']
        update_transforms = []
        if update_data.get('transforms'):
            update_transforms = update_data['transforms']
        print('Updating source ' + os.path.dirname(source) + ' from ' + update_url)
        try:
            updated_file = get_file_by_url(update_url)
            for transform in update_transforms:
                updated_file = transform_methods[transform](updated_file)
            updated_file = updated_file.replace('\r', '')
            hosts_file = open(path_join_robust(BASEDIR_PATH, os.path.dirname(source), host_filename), 'wb')
            write_data(hosts_file, updated_file)
            hosts_file.close()
        except Exception:
            print('Error in updating source: ', update_url)

def create_initial_file(**initial_file_params):
    if False:
        return 10
    '\n    Initialize the file in which we merge all host files for later pruning.\n\n    Parameters\n    ----------\n    header_params : kwargs\n        Dictionary providing additional parameters for populating the initial file\n        information. Currently, those fields are:\n\n        1) nounifiedhosts\n    '
    merge_file = tempfile.NamedTemporaryFile()
    if not initial_file_params['nounifiedhosts']:
        for source in sort_sources(recursive_glob(settings['datapath'], settings['hostfilename'])):
            start = '# Start {}\n\n'.format(os.path.basename(os.path.dirname(source)))
            end = '\n# End {}\n\n'.format(os.path.basename(os.path.dirname(source)))
            with open(source, 'r', encoding='UTF-8') as curFile:
                write_data(merge_file, start + curFile.read() + end)
    for source in settings['extensions']:
        for filename in sort_sources(recursive_glob(path_join_robust(settings['extensionspath'], source), settings['hostfilename'])):
            with open(filename, 'r') as curFile:
                write_data(merge_file, curFile.read())
    maybe_copy_example_file(settings['blacklistfile'])
    if os.path.isfile(settings['blacklistfile']):
        with open(settings['blacklistfile'], 'r') as curFile:
            write_data(merge_file, curFile.read())
    return merge_file

def compress_file(input_file, target_ip, output_file):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reduce the file dimension removing non-necessary lines (empty lines and\n    comments) and putting multiple domains in each line.\n    Reducing the number of lines of the file, the parsing under Microsoft\n    Windows is much faster.\n\n    Parameters\n    ----------\n    input_file : file\n        The file object that contains the hostnames that we are reducing.\n    target_ip : str\n        The target IP address.\n    output_file : file\n        The file object that will contain the reduced hostnames.\n    '
    input_file.seek(0)
    write_data(output_file, '\n')
    target_ip_len = len(target_ip)
    lines = [target_ip]
    lines_index = 0
    for line in input_file.readlines():
        line = line.decode('UTF-8')
        if line.startswith(target_ip):
            if lines[lines_index].count(' ') < 9:
                lines[lines_index] += ' ' + line[target_ip_len:line.find('#')].strip()
            else:
                lines[lines_index] += '\n'
                lines.append(line[:line.find('#')].strip())
                lines_index += 1
    for line in lines:
        write_data(output_file, line)
    input_file.close()

def minimise_file(input_file, target_ip, output_file):
    if False:
        print('Hello World!')
    '\n    Reduce the file dimension removing non-necessary lines (empty lines and\n    comments).\n\n    Parameters\n    ----------\n    input_file : file\n        The file object that contains the hostnames that we are reducing.\n    target_ip : str\n        The target IP address.\n    output_file : file\n        The file object that will contain the reduced hostnames.\n    '
    input_file.seek(0)
    write_data(output_file, '\n')
    lines = []
    for line in input_file.readlines():
        line = line.decode('UTF-8')
        if line.startswith(target_ip):
            lines.append(line[:line.find('#')].strip() + '\n')
    for line in lines:
        write_data(output_file, line)
    input_file.close()

def remove_dups_and_excl(merge_file, exclusion_regexes, output_file=None):
    if False:
        print('Hello World!')
    '\n    Remove duplicates and remove hosts that we are excluding.\n\n    We check for duplicate hostnames as well as remove any hostnames that\n    have been explicitly excluded by the user.\n\n    Parameters\n    ----------\n    merge_file : file\n        The file object that contains the hostnames that we are pruning.\n    exclusion_regexes : list\n        The list of regex patterns used to exclude domains.\n    output_file : file\n        The file object in which the result is written. If None, the file\n        \'settings["outputpath"]\' will be created.\n    '
    number_of_rules = settings['numberofrules']
    maybe_copy_example_file(settings['whitelistfile'])
    if os.path.isfile(settings['whitelistfile']):
        with open(settings['whitelistfile'], 'r') as ins:
            for line in ins:
                line = line.strip(' \t\n\r')
                if line and (not line.startswith('#')):
                    settings['exclusions'].append(line)
    if not os.path.exists(settings['outputpath']):
        os.makedirs(settings['outputpath'])
    if output_file is None:
        final_file = open(path_join_robust(settings['outputpath'], 'hosts'), 'w+b')
    else:
        final_file = output_file
    merge_file.seek(0)
    hostnames = {'localhost', 'localhost.localdomain', 'local', 'broadcasthost'}
    exclusions = settings['exclusions']
    for line in merge_file.readlines():
        write_line = True
        line = line.decode('UTF-8')
        line = line.replace('\t+', ' ')
        line = line.rstrip(' .')
        if line[0] == '#' or re.match('^\\s*$', line[0]):
            write_data(final_file, line)
            continue
        if '::1' in line:
            continue
        stripped_rule = strip_rule(line)
        if not stripped_rule or matches_exclusions(stripped_rule, exclusion_regexes):
            continue
        if '@' in stripped_rule:
            continue
        (hostname, normalized_rule) = normalize_rule(stripped_rule, target_ip=settings['targetip'], keep_domain_comments=settings['keepdomaincomments'])
        for exclude in exclusions:
            if re.search('(^|[\\s\\.])' + re.escape(exclude) + '\\s', line):
                write_line = False
                break
        if normalized_rule and hostname not in hostnames and write_line:
            write_data(final_file, normalized_rule)
            hostnames.add(hostname)
            number_of_rules += 1
    settings['numberofrules'] = number_of_rules
    merge_file.close()
    if output_file is None:
        return final_file

def normalize_rule(rule, target_ip, keep_domain_comments):
    if False:
        print('Hello World!')
    '\n    Standardize and format the rule string provided.\n\n    Parameters\n    ----------\n    rule : str\n        The rule whose spelling and spacing we are standardizing.\n    target_ip : str\n        The target IP address for the rule.\n    keep_domain_comments : bool\n        Whether or not to keep comments regarding these domains in\n        the normalized rule.\n\n    Returns\n    -------\n    normalized_rule : tuple\n        A tuple of the hostname and the rule string with spelling\n        and spacing reformatted.\n    '

    def normalize_response(extracted_hostname: str, extracted_suffix: Optional[str]) -> Tuple[str, str]:
        if False:
            while True:
                i = 10
        '\n        Normalizes the responses after the provision of the extracted\n        hostname and suffix - if exist.\n\n        Parameters\n        ----------\n        extracted_hostname: str\n            The extracted hostname to work with.\n        extracted_suffix: str\n            The extracted suffix to with.\n\n        Returns\n        -------\n        normalized_response: tuple\n            A tuple of the hostname and the rule string with spelling\n            and spacing reformatted.\n        '
        rule = '%s %s' % (target_ip, extracted_hostname)
        if keep_domain_comments and extracted_suffix:
            if not extracted_suffix.strip().startswith('#'):
                rule += ' # %s' % extracted_suffix
            else:
                rule += ' %s' % extracted_suffix
        return (extracted_hostname, rule + '\n')

    def is_ip(dataset: str) -> bool:
        if False:
            return 10
        '\n        Checks whether the given dataset is an IP.\n\n        Parameters\n        ----------\n\n        dataset: str\n            The dataset to work with.\n\n        Returns\n        -------\n        is_ip: bool\n            Whether the dataset is an IP.\n        '
        try:
            _ = ipaddress.ip_address(dataset)
            return True
        except ValueError:
            return False

    def belch_unwanted(unwanted: str) -> Tuple[None, None]:
        if False:
            while True:
                i = 10
        '\n        Belches unwanted to screen.\n\n        Parameters\n        ----------\n        unwanted: str\n            The unwanted string to belch.\n\n        Returns\n        -------\n        belched: tuple\n            A tuple of None, None.\n        '
        '\n        finally, if we get here, just belch to screen\n        '
        print('==>%s<==' % unwanted)
        return (None, None)
    '\n    first try: IP followed by domain\n    '
    static_ip_regex = '^(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})$'
    split_rule = rule.split(maxsplit=1)
    if is_ip(split_rule[0]):
        if ' ' or '\t' in split_rule[-1]:
            try:
                (hostname, suffix) = split_rule[-1].split(maxsplit=1)
            except ValueError:
                (hostname, suffix) = (split_rule[-1], None)
        else:
            (hostname, suffix) = (split_rule[-1], None)
        hostname = hostname.lower()
        if is_ip(hostname) or re.search(static_ip_regex, hostname) or '.' not in hostname or ('/' in hostname) or ('..' in hostname) or (':' in hostname):
            return belch_unwanted(rule)
        return normalize_response(hostname, suffix)
    if not re.search(static_ip_regex, split_rule[0]) and ':' not in split_rule[0] and ('..' not in split_rule[0]) and ('/' not in split_rule[0]) and ('.' in split_rule[0]):
        try:
            (hostname, suffix) = split_rule
        except ValueError:
            (hostname, suffix) = (split_rule[0], None)
        hostname = hostname.lower()
        return normalize_response(hostname, suffix)
    return belch_unwanted(rule)

def strip_rule(line):
    if False:
        while True:
            i = 10
    '\n    Sanitize a rule string provided before writing it to the output hosts file.\n\n    Parameters\n    ----------\n    line : str\n        The rule provided for sanitation.\n\n    Returns\n    -------\n    sanitized_line : str\n        The sanitized rule.\n    '
    return ' '.join(line.split())

def write_opening_header(final_file, **header_params):
    if False:
        print('Hello World!')
    '\n    Write the header information into the newly-created hosts file.\n\n    Parameters\n    ----------\n    final_file : file\n        The file object that points to the newly-created hosts file.\n    header_params : kwargs\n        Dictionary providing additional parameters for populating the header\n        information. Currently, those fields are:\n\n        1) extensions\n        2) numberofrules\n        3) outputsubfolder\n        4) skipstatichosts\n        5) nounifiedhosts\n    '
    final_file.seek(0)
    file_contents = final_file.read()
    final_file.seek(0)
    no_unified_hosts = header_params['nounifiedhosts']
    if header_params['extensions']:
        if no_unified_hosts:
            if len(header_params['extensions']) > 1:
                write_data(final_file, '# Title: StevenBlack/hosts extensions {0} and {1} \n#\n'.format(', '.join(header_params['extensions'][:-1]), header_params['extensions'][-1]))
            else:
                write_data(final_file, '# Title: StevenBlack/hosts extension {0}\n#\n'.format(', '.join(header_params['extensions'])))
        elif len(header_params['extensions']) > 1:
            write_data(final_file, '# Title: StevenBlack/hosts with the {0} and {1} extensions\n#\n'.format(', '.join(header_params['extensions'][:-1]), header_params['extensions'][-1]))
        else:
            write_data(final_file, '# Title: StevenBlack/hosts with the {0} extension\n#\n'.format(', '.join(header_params['extensions'])))
    else:
        write_data(final_file, '# Title: StevenBlack/hosts\n#\n')
    write_data(final_file, '# This hosts file is a merged collection of hosts from reputable sources,\n')
    write_data(final_file, '# with a dash of crowd sourcing via GitHub\n#\n')
    write_data(final_file, '# Date: ' + time.strftime('%d %B %Y %H:%M:%S (%Z)', time.gmtime()) + '\n')
    if header_params['extensions']:
        if header_params['nounifiedhosts']:
            write_data(final_file, '# The unified hosts file was not used while generating this file.\n# Extensions used to generate this file: ' + ', '.join(header_params['extensions']) + '\n')
        else:
            write_data(final_file, '# Extensions added to this file: ' + ', '.join(header_params['extensions']) + '\n')
    write_data(final_file, '# Number of unique domains: {:,}\n#\n'.format(header_params['numberofrules']))
    write_data(final_file, '# Fetch the latest version of this file: https://raw.githubusercontent.com/StevenBlack/hosts/master/' + path_join_robust(header_params['outputsubfolder'], '').replace('\\', '/') + 'hosts\n')
    write_data(final_file, '# Project home page: https://github.com/StevenBlack/hosts\n')
    write_data(final_file, '# Project releases: https://github.com/StevenBlack/hosts/releases\n#\n')
    write_data(final_file, '# ===============================================================\n')
    write_data(final_file, '\n')
    if not header_params['skipstatichosts']:
        write_data(final_file, '127.0.0.1 localhost\n')
        write_data(final_file, '127.0.0.1 localhost.localdomain\n')
        write_data(final_file, '127.0.0.1 local\n')
        write_data(final_file, '255.255.255.255 broadcasthost\n')
        write_data(final_file, '::1 localhost\n')
        write_data(final_file, '::1 ip6-localhost\n')
        write_data(final_file, '::1 ip6-loopback\n')
        write_data(final_file, 'fe80::1%lo0 localhost\n')
        write_data(final_file, 'ff00::0 ip6-localnet\n')
        write_data(final_file, 'ff00::0 ip6-mcastprefix\n')
        write_data(final_file, 'ff02::1 ip6-allnodes\n')
        write_data(final_file, 'ff02::2 ip6-allrouters\n')
        write_data(final_file, 'ff02::3 ip6-allhosts\n')
        write_data(final_file, '0.0.0.0 0.0.0.0\n')
        if platform.system() == 'Linux':
            write_data(final_file, '127.0.1.1 ' + socket.gethostname() + '\n')
            write_data(final_file, '127.0.0.53 ' + socket.gethostname() + '\n')
        write_data(final_file, '\n')
    preamble = path_join_robust(BASEDIR_PATH, 'myhosts')
    maybe_copy_example_file(preamble)
    if os.path.isfile(preamble):
        with open(preamble, 'r') as f:
            write_data(final_file, f.read())
    final_file.write(file_contents)

def update_readme_data(readme_file, **readme_updates):
    if False:
        return 10
    '\n    Update the host and website information provided in the README JSON data.\n\n    Parameters\n    ----------\n    readme_file : str\n        The name of the README file to update.\n    readme_updates : kwargs\n        Dictionary providing additional JSON fields to update before\n        saving the data. Currently, those fields are:\n\n        1) extensions\n        2) sourcesdata\n        3) numberofrules\n        4) outputsubfolder\n        5) nounifiedhosts\n    '
    extensions_key = 'base'
    extensions = readme_updates['extensions']
    no_unified_hosts = readme_updates['nounifiedhosts']
    if extensions:
        extensions_key = '-'.join(extensions)
        if no_unified_hosts:
            extensions_key = extensions_key + '-only'
    output_folder = readme_updates['outputsubfolder']
    generation_data = {'location': path_join_robust(output_folder, ''), 'no_unified_hosts': no_unified_hosts, 'entries': readme_updates['numberofrules'], 'sourcesdata': readme_updates['sourcesdata']}
    with open(readme_file, 'r') as f:
        readme_data = json.load(f)
        readme_data[extensions_key] = generation_data
    for (denomination, data) in readme_data.copy().items():
        if 'location' in data and data['location'] and ('\\' in data['location']):
            readme_data[denomination]['location'] = data['location'].replace('\\', '/')
    with open(readme_file, 'w') as f:
        json.dump(readme_data, f)

def move_hosts_file_into_place(final_file):
    if False:
        while True:
            i = 10
    '\n    Move the newly-created hosts file into its correct location on the OS.\n\n    For UNIX systems, the hosts file is "etc/hosts." On Windows, it\'s\n    "C:\\Windows\\System32\\drivers\\etc\\hosts."\n\n    For this move to work, you must have administrator privileges to do this.\n    On UNIX systems, this means having "sudo" access, and on Windows, it\n    means being able to run command prompt in administrator mode.\n\n    Parameters\n    ----------\n    final_file : file object\n        The newly-created hosts file to move.\n    '
    filename = os.path.abspath(final_file.name)
    try:
        if not Path(filename).exists():
            raise FileNotFoundError
    except Exception:
        print_failure(f'{filename} does not exist.')
        return False
    if platform.system() == 'Windows':
        target_file = str(Path(os.getenv('SystemRoot')) / 'system32' / 'drivers' / 'etc' / 'hosts')
    else:
        target_file = '/etc/hosts'
    if os.getenv('IN_CONTAINER'):
        print(f'Running in container, so we will replace the content of {target_file}.')
        try:
            with open(target_file, 'w') as target_stream:
                with open(filename, 'r') as source_stream:
                    source = source_stream.read()
                    target_stream.write(source)
            return True
        except Exception:
            print_failure(f'Replacing content of {target_file} failed.')
            return False
    elif platform.system() == 'Linux' or platform.system() == 'Windows' or platform.system() == 'Darwin':
        print(f'Replacing {target_file} requires root privileges. You might need to enter your password.')
        try:
            subprocess.run(SUDO + ['cp', filename, target_file], check=True)
            return True
        except subprocess.CalledProcessError:
            print_failure(f'Replacing {target_file} failed.')
            return False

def flush_dns_cache():
    if False:
        for i in range(10):
            print('nop')
    '\n    Flush the DNS cache.\n    '
    print('Flushing the DNS cache to utilize new hosts file...')
    print('Flushing the DNS cache requires administrative privileges. You might need to enter your password.')
    dns_cache_found = False
    if platform.system() == 'Darwin':
        if subprocess.call(SUDO + ['killall', '-HUP', 'mDNSResponder']):
            print_failure('Flushing the DNS cache failed.')
    elif os.name == 'nt':
        print('Automatically flushing the DNS cache is not yet supported.')
        print("Please copy and paste the command 'ipconfig /flushdns' in administrator command prompt after running this script.")
    else:
        nscd_prefixes = ['/etc', '/etc/rc.d']
        nscd_msg = 'Flushing the DNS cache by restarting nscd {result}'
        for nscd_prefix in nscd_prefixes:
            nscd_cache = nscd_prefix + '/init.d/nscd'
            if os.path.isfile(nscd_cache):
                dns_cache_found = True
                if subprocess.call(SUDO + [nscd_cache, 'restart']):
                    print_failure(nscd_msg.format(result='failed'))
                else:
                    print_success(nscd_msg.format(result='succeeded'))
        centos_file = '/etc/init.d/network'
        centos_msg = 'Flushing the DNS cache by restarting network {result}'
        if os.path.isfile(centos_file):
            if subprocess.call(SUDO + [centos_file, 'restart']):
                print_failure(centos_msg.format(result='failed'))
            else:
                print_success(centos_msg.format(result='succeeded'))
        system_prefixes = ['/usr', '']
        service_types = ['NetworkManager', 'wicd', 'dnsmasq', 'networking']
        restarted_services = []
        for system_prefix in system_prefixes:
            systemctl = system_prefix + '/bin/systemctl'
            system_dir = system_prefix + '/lib/systemd/system'
            for service_type in service_types:
                service = service_type + '.service'
                if service in restarted_services:
                    continue
                service_file = path_join_robust(system_dir, service)
                service_msg = 'Flushing the DNS cache by restarting ' + service + ' {result}'
                if os.path.isfile(service_file):
                    if 0 != subprocess.call([systemctl, 'status', service], stdout=subprocess.DEVNULL):
                        continue
                    dns_cache_found = True
                    if subprocess.call(SUDO + [systemctl, 'restart', service]):
                        print_failure(service_msg.format(result='failed'))
                    else:
                        print_success(service_msg.format(result='succeeded'))
                    restarted_services.append(service)
        dns_clean_file = '/etc/init.d/dns-clean'
        dns_clean_msg = 'Flushing the DNS cache via dns-clean executable {result}'
        if os.path.isfile(dns_clean_file):
            dns_cache_found = True
            if subprocess.call(SUDO + [dns_clean_file, 'start']):
                print_failure(dns_clean_msg.format(result='failed'))
            else:
                print_success(dns_clean_msg.format(result='succeeded'))
        if not dns_cache_found:
            print_failure('Unable to determine DNS management tool.')

def remove_old_hosts_file(path_to_file, file_name, backup):
    if False:
        print('Hello World!')
    '\n    Remove the old hosts file.\n\n    This is a hotfix because merging with an already existing hosts file leads\n    to artifacts and duplicates.\n\n    Parameters\n    ----------\n    backup : boolean, default False\n        Whether or not to backup the existing hosts file.\n    '
    full_file_path = path_join_robust(path_to_file, file_name)
    if os.path.exists(full_file_path):
        if backup:
            backup_file_path = full_file_path + '-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
            shutil.copy(full_file_path, backup_file_path)
        os.remove(full_file_path)
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
    open(full_file_path, 'a').close()

def domain_to_idna(line):
    if False:
        i = 10
        return i + 15
    "\n    Encode a domain that is present into a line into `idna`. This way we\n    avoid most encoding issues.\n\n    Parameters\n    ----------\n    line : str\n        The line we have to encode/decode.\n\n    Returns\n    -------\n    line : str\n        The line in a converted format.\n\n    Notes\n    -----\n    - This function encodes only the domain to `idna` format because in\n        most cases, the encoding issue is due to a domain which looks like\n        `b'É¢oogle.com'.decode('idna')`.\n    - About the splitting:\n        We split because we only want to encode the domain and not the full\n        line, which may cause some issues. Keep in mind that we split, but we\n        still concatenate once we encoded the domain.\n\n        - The following split the prefix `0.0.0.0` or `127.0.0.1` of a line.\n        - The following also split the trailing comment of a given line.\n    "
    if not line.startswith('#'):
        tabs = '\t'
        space = ' '
        (tabs_position, space_position) = (line.find(tabs), line.find(space))
        if tabs_position > -1 and space_position > -1:
            if space_position < tabs_position:
                separator = space
            else:
                separator = tabs
        elif not tabs_position == -1:
            separator = tabs
        elif not space_position == -1:
            separator = space
        else:
            separator = ''
        if separator:
            splited_line = line.split(separator)
            try:
                index = 1
                while index < len(splited_line):
                    if splited_line[index]:
                        break
                    index += 1
                if '#' in splited_line[index]:
                    index_comment = splited_line[index].find('#')
                    if index_comment > -1:
                        comment = splited_line[index][index_comment:]
                        splited_line[index] = splited_line[index].split(comment)[0].encode('IDNA').decode('UTF-8') + comment
                splited_line[index] = splited_line[index].encode('IDNA').decode('UTF-8')
            except IndexError:
                pass
            return separator.join(splited_line)
        return line.encode('IDNA').decode('UTF-8')
    return line.encode('UTF-8').decode('UTF-8')

def maybe_copy_example_file(file_path):
    if False:
        return 10
    '\n    Given a file path, copy over its ".example" if the path doesn\'t exist.\n\n    If the path does exist, nothing happens in this function.\n\n    If the path doesn\'t exist, and the ".example" file doesn\'t exist, nothing happens in this function.\n\n    Parameters\n    ----------\n    file_path : str\n        The full file path to check.\n    '
    if not os.path.isfile(file_path):
        example_file_path = file_path + '.example'
        if os.path.isfile(example_file_path):
            shutil.copyfile(example_file_path, file_path)

def get_file_by_url(url, params=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Retrieve the contents of the hosts file at the URL, then pass it through domain_to_idna().\n\n    Parameters are passed to the requests.get() function.\n\n    Parameters\n    ----------\n    url : str or bytes\n        URL for the new Request object.\n    params :\n        Dictionary, list of tuples or bytes to send in the query string for the Request.\n    kwargs :\n        Optional arguments that request takes.\n\n    Returns\n    -------\n    url_data : str or None\n        The data retrieved at that URL from the file. Returns None if the\n        attempted retrieval is unsuccessful.\n    '
    try:
        req = requests.get(url=url, params=params, **kwargs)
    except requests.exceptions.RequestException:
        print('Error retrieving data from {}'.format(url))
        return None
    req.encoding = req.apparent_encoding
    res_text = '\n'.join([domain_to_idna(line) for line in req.text.split('\n')])
    return res_text

def write_data(f, data):
    if False:
        while True:
            i = 10
    '\n    Write data to a file object.\n\n    Parameters\n    ----------\n    f : file\n        The file object at which to write the data.\n    data : str\n        The data to write to the file.\n    '
    f.write(bytes(data, 'UTF-8'))

def list_dir_no_hidden(path):
    if False:
        return 10
    '\n    List all files in a directory, except for hidden files.\n\n    Parameters\n    ----------\n    path : str\n        The path of the directory whose files we wish to list.\n    '
    return glob(os.path.join(path, '*'))

def query_yes_no(question, default='yes'):
    if False:
        i = 10
        return i + 15
    '\n    Ask a yes/no question via input() and get answer from the user.\n\n    Inspired by the following implementation:\n\n    https://code.activestate.com/recipes/577058/\n\n    Parameters\n    ----------\n    question : str\n        The question presented to the user.\n    default : str, default "yes"\n        The presumed answer if the user just hits <Enter>. It must be "yes",\n        "no", or None (means an answer is required of the user).\n\n    Returns\n    -------\n    yes : Whether or not the user replied yes to the question.\n    '
    valid = {'yes': 'yes', 'y': 'yes', 'ye': 'yes', 'no': 'no', 'n': 'no'}
    prompt = {None: ' [y/n] ', 'yes': ' [Y/n] ', 'no': ' [y/N] '}.get(default, None)
    if not prompt:
        raise ValueError("invalid default answer: '%s'" % default)
    reply = None
    while not reply:
        sys.stdout.write(colorize(question, Colors.PROMPT) + prompt)
        choice = input().lower()
        reply = None
        if default and (not choice):
            reply = default
        elif choice in valid:
            reply = valid[choice]
        else:
            print_failure("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
    return reply == 'yes'

def is_valid_user_provided_domain_format(domain):
    if False:
        while True:
            i = 10
    '\n    Check whether a provided domain is valid.\n\n    Parameters\n    ----------\n    domain : str\n        The domain against which to check.\n\n    Returns\n    -------\n    valid_domain : bool\n        Whether or not the domain provided is valid.\n    '
    if domain == '':
        print("You didn't enter a domain. Try again.")
        return False
    domain_regex = re.compile('www\\d{0,3}[.]|https?')
    if domain_regex.match(domain):
        print('The domain ' + domain + ' is not valid. Do not include www.domain.com or http(s)://domain.com. Try again.')
        return False
    else:
        return True

def recursive_glob(stem, file_pattern):
    if False:
        while True:
            i = 10
    '\n    Recursively match files in a directory according to a pattern.\n\n    Parameters\n    ----------\n    stem : str\n        The directory in which to recurse\n    file_pattern : str\n        The filename regex pattern to which to match.\n\n    Returns\n    -------\n    matches_list : list\n        A list of filenames in the directory that match the file pattern.\n    '
    if sys.version_info >= (3, 5):
        return glob(stem + '/**/' + file_pattern, recursive=True)
    else:
        if stem == str('*'):
            stem = '.'
        matches = []
        for (root, dirnames, filenames) in os.walk(stem):
            for filename in fnmatch.filter(filenames, file_pattern):
                matches.append(path_join_robust(root, filename))
    return matches

def path_join_robust(path, *paths):
    if False:
        while True:
            i = 10
    '\n    Wrapper around `os.path.join` with handling for locale issues.\n\n    Parameters\n    ----------\n    path : str\n        The first path to join.\n    paths : varargs\n        Subsequent path strings to join.\n\n    Returns\n    -------\n    joined_path : str\n        The joined path string of the two path inputs.\n\n    Raises\n    ------\n    locale.Error : A locale issue was detected that prevents path joining.\n    '
    try:
        path = str(path)
        paths = [str(another_path) for another_path in paths]
        return os.path.join(path, *paths)
    except UnicodeDecodeError as e:
        raise locale.Error('Unable to construct path. This is likely a LOCALE issue:\n\n' + str(e))

class Colors(object):
    PROMPT = '\x1b[94m'
    SUCCESS = '\x1b[92m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'

def supports_color():
    if False:
        print('Hello World!')
    '\n    Check whether the running terminal or command prompt supports color.\n\n    Inspired by StackOverflow link (and Django implementation) here:\n\n    https://stackoverflow.com/questions/7445658\n\n    Returns\n    -------\n    colors_supported : bool\n        Whether the running terminal or command prompt supports color.\n    '
    sys_platform = sys.platform
    supported = sys_platform != 'Pocket PC' and (sys_platform != 'win32' or 'ANSICON' in os.environ)
    atty_connected = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return supported and atty_connected

def colorize(text, color):
    if False:
        print('Hello World!')
    '\n    Wrap a string so that it displays in a particular color.\n\n    This function adds a prefix and suffix to a text string so that it is\n    displayed as a particular color, either in command prompt or the terminal.\n\n    If the running terminal or command prompt does not support color, the\n    original text is returned without being wrapped.\n\n    Parameters\n    ----------\n    text : str\n        The message to display.\n    color : str\n        The color string prefix to put before the text.\n\n    Returns\n    -------\n    wrapped_str : str\n        The wrapped string to display in color, if possible.\n    '
    if not supports_color():
        return text
    return color + text + Colors.ENDC

def print_success(text):
    if False:
        while True:
            i = 10
    '\n    Print a success message.\n\n    Parameters\n    ----------\n    text : str\n        The message to display.\n    '
    print(colorize(text, Colors.SUCCESS))

def print_failure(text):
    if False:
        i = 10
        return i + 15
    '\n    Print a failure message.\n\n    Parameters\n    ----------\n    text : str\n        The message to display.\n    '
    print(colorize(text, Colors.FAIL))
if __name__ == '__main__':
    main()