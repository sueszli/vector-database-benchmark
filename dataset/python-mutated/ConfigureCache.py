"""Show or convert the configuration of an SCons cache directory.

A cache of derived files is stored by file signature.
The files are split into directories named by the first few
digits of the signature. The prefix length used for directory
names can be changed by this script.
"""
import argparse
import glob
import json
import os

def rearrange_cache_entries(current_prefix_len, new_prefix_len):
    if False:
        while True:
            i = 10
    'Move cache files if prefix length changed.\n\n    Move the existing cache files to new directories of the\n    appropriate name length and clean up the old directories.\n    '
    print('Changing prefix length from', current_prefix_len, 'to', new_prefix_len)
    dirs = set()
    old_dirs = set()
    for file in glob.iglob(os.path.join('*', '*')):
        name = os.path.basename(file)
        dname = name[:current_prefix_len].upper()
        if dname not in old_dirs:
            print('Migrating', dname)
            old_dirs.add(dname)
        dname = name[:new_prefix_len].upper()
        if dname not in dirs:
            os.mkdir(dname)
            dirs.add(dname)
        os.rename(file, os.path.join(dname, name))
    for dname in old_dirs:
        os.rmdir(dname)
config_entries = {'prefix_len': {'implicit': 1, 'default': 2, 'command-line': {'help': 'Length of cache file name used as subdirectory prefix', 'metavar': '<number>', 'type': int}, 'converter': rearrange_cache_entries}}

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Modify the configuration of an scons cache directory', epilog='\n               Unspecified options will not be changed unless they are not\n               set at all, in which case they are set to an appropriate default.\n               ')
    parser.add_argument('cache-dir', help='Path to scons cache directory')
    for param in config_entries:
        parser.add_argument('--' + param.replace('_', '-'), **config_entries[param]['command-line'])
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('--show', action='store_true', help='show current configuration')
    args = dict([x for x in vars(parser.parse_args()).items() if x[1]])
    cache = args['cache-dir']
    if not os.path.isdir(cache):
        raise RuntimeError('There is no cache directory named %s' % cache)
    os.chdir(cache)
    del args['cache-dir']
    if not os.path.exists('config'):
        expected = ['{:X}'.format(x) for x in range(0, 16)]
        if not set(os.listdir('.')).issubset(expected):
            raise RuntimeError('%s does not look like a valid version 1 cache directory' % cache)
        config = dict()
    else:
        with open('config') as conf:
            config = json.load(conf)
    if args.get('show', None):
        print("Current configuration in '%s':" % cache)
        print(json.dumps(config, sort_keys=True, indent=4, separators=(',', ': ')))
        file_count = 0
        for (_, _, files) in os.walk('.'):
            file_count += len(files)
        if file_count:
            file_count -= 1
        print('Cache contains %s files' % file_count)
        del args['show']
    for key in config_entries:
        if key not in config:
            if 'implicit' in config_entries[key]:
                config[key] = config_entries[key]['implicit']
            else:
                config[key] = config_entries[key]['default']
            if key not in args:
                args[key] = config_entries[key]['default']
    for key in args:
        if args[key] != config[key]:
            if 'converter' in config_entries[key]:
                config_entries[key]['converter'](config[key], args[key])
            config[key] = args[key]
    with open('config', 'w') as conf:
        json.dump(config, conf)
if __name__ == '__main__':
    main()