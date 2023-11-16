import os
from argparse import ArgumentParser
from subprocess import run
import django
from django.conf import settings
from django.core.management import call_command
HAVE_JS = ['admin']

def _get_locale_dirs(resources, include_core=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a tuple (contrib name, absolute path) for all locale directories,\n    optionally including the django core catalog.\n    If resources list is not None, filter directories matching resources content.\n    '
    contrib_dir = os.path.join(os.getcwd(), 'django', 'contrib')
    dirs = []
    for contrib_name in os.listdir(contrib_dir):
        path = os.path.join(contrib_dir, contrib_name, 'locale')
        if os.path.isdir(path):
            dirs.append((contrib_name, path))
            if contrib_name in HAVE_JS:
                dirs.append(('%s-js' % contrib_name, path))
    if include_core:
        dirs.insert(0, ('core', os.path.join(os.getcwd(), 'django', 'conf', 'locale')))
    if resources is not None:
        res_names = [d[0] for d in dirs]
        dirs = [ld for ld in dirs if ld[0] in resources]
        if len(resources) > len(dirs):
            print('You have specified some unknown resources. Available resource names are: %s' % (', '.join(res_names),))
            exit(1)
    return dirs

def _tx_resource_for_name(name):
    if False:
        i = 10
        return i + 15
    'Return the Transifex resource name'
    if name == 'core':
        return 'django.core'
    else:
        return 'django.contrib-%s' % name

def _check_diff(cat_name, base_path):
    if False:
        return 10
    '\n    Output the approximate number of changed/added strings in the en catalog.\n    '
    po_path = '%(path)s/en/LC_MESSAGES/django%(ext)s.po' % {'path': base_path, 'ext': 'js' if cat_name.endswith('-js') else ''}
    p = run("git diff -U0 %s | egrep '^[-+]msgid' | wc -l" % po_path, capture_output=True, shell=True)
    num_changes = int(p.stdout.strip())
    print("%d changed/added messages in '%s' catalog." % (num_changes, cat_name))

def update_catalogs(resources=None, languages=None):
    if False:
        return 10
    '\n    Update the en/LC_MESSAGES/django.po (main and contrib) files with\n    new/updated translatable strings.\n    '
    settings.configure()
    django.setup()
    if resources is not None:
        print('`update_catalogs` will always process all resources.')
    contrib_dirs = _get_locale_dirs(None, include_core=False)
    os.chdir(os.path.join(os.getcwd(), 'django'))
    print('Updating en catalogs for Django and contrib apps...')
    call_command('makemessages', locale=['en'])
    print('Updating en JS catalogs for Django and contrib apps...')
    call_command('makemessages', locale=['en'], domain='djangojs')
    _check_diff('core', os.path.join(os.getcwd(), 'conf', 'locale'))
    for (name, dir_) in contrib_dirs:
        _check_diff(name, dir_)

def lang_stats(resources=None, languages=None):
    if False:
        while True:
            i = 10
    "\n    Output language statistics of committed translation files for each\n    Django catalog.\n    If resources is provided, it should be a list of translation resource to\n    limit the output (e.g. ['core', 'gis']).\n    "
    locale_dirs = _get_locale_dirs(resources)
    for (name, dir_) in locale_dirs:
        print("\nShowing translations stats for '%s':" % name)
        langs = sorted((d for d in os.listdir(dir_) if not d.startswith('_')))
        for lang in langs:
            if languages and lang not in languages:
                continue
            po_path = '{path}/{lang}/LC_MESSAGES/django{ext}.po'.format(path=dir_, lang=lang, ext='js' if name.endswith('-js') else '')
            p = run(['msgfmt', '-vc', '-o', '/dev/null', po_path], capture_output=True, env={'LANG': 'C'}, encoding='utf-8')
            if p.returncode == 0:
                print('%s: %s' % (lang, p.stderr.strip()))
            else:
                print('Errors happened when checking %s translation for %s:\n%s' % (lang, name, p.stderr))

def fetch(resources=None, languages=None):
    if False:
        print('Hello World!')
    '\n    Fetch translations from Transifex, wrap long lines, generate mo files.\n    '
    locale_dirs = _get_locale_dirs(resources)
    errors = []
    for (name, dir_) in locale_dirs:
        if languages is None:
            run(['tx', 'pull', '-r', _tx_resource_for_name(name), '-a', '-f', '--minimum-perc=5'])
            target_langs = sorted((d for d in os.listdir(dir_) if not d.startswith('_') and d != 'en'))
        else:
            for lang in languages:
                run(['tx', 'pull', '-r', _tx_resource_for_name(name), '-f', '-l', lang])
            target_langs = languages
        for lang in target_langs:
            po_path = '%(path)s/%(lang)s/LC_MESSAGES/django%(ext)s.po' % {'path': dir_, 'lang': lang, 'ext': 'js' if name.endswith('-js') else ''}
            if not os.path.exists(po_path):
                print('No %(lang)s translation for resource %(name)s' % {'lang': lang, 'name': name})
                continue
            run(['msgcat', '--no-location', '-o', po_path, po_path])
            msgfmt = run(['msgfmt', '-c', '-o', '%s.mo' % po_path[:-3], po_path])
            if msgfmt.returncode != 0:
                errors.append((name, lang))
    if errors:
        print('\nWARNING: Errors have occurred in following cases:')
        for (resource, lang) in errors:
            print('\tResource %s for language %s' % (resource, lang))
        exit(1)
if __name__ == '__main__':
    RUNABLE_SCRIPTS = ('update_catalogs', 'lang_stats', 'fetch')
    parser = ArgumentParser()
    parser.add_argument('cmd', nargs=1, choices=RUNABLE_SCRIPTS)
    parser.add_argument('-r', '--resources', action='append', help='limit operation to the specified resources')
    parser.add_argument('-l', '--languages', action='append', help='limit operation to the specified languages')
    options = parser.parse_args()
    eval(options.cmd[0])(options.resources, options.languages)