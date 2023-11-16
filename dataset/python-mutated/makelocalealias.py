"""
    Convert the X11 locale.alias file into a mapping dictionary suitable
    for locale.py.

    Written by Marc-Andre Lemburg <mal@genix.com>, 2004-12-10.

"""
import locale
import sys
_locale = locale
LOCALE_ALIAS = '/usr/share/X11/locale/locale.alias'
SUPPORTED = '/usr/share/i18n/SUPPORTED'

def parse(filename):
    if False:
        while True:
            i = 10
    with open(filename, encoding='latin1') as f:
        lines = list(f)
    lines = [line for line in lines if 'ï¿½' not in line]
    data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[:1] == '#':
            continue
        (locale, alias) = line.split()
        if '@' in alias:
            (alias_lang, _, alias_mod) = alias.partition('@')
            if '.' in alias_mod:
                (alias_mod, _, alias_enc) = alias_mod.partition('.')
                alias = alias_lang + '.' + alias_enc + '@' + alias_mod
        if locale[-1] == ':':
            locale = locale[:-1]
        locale = locale.lower()
        if len(locale) == 1 and locale != 'c':
            continue
        if '.' in locale:
            (lang, encoding) = locale.split('.')[:2]
            encoding = encoding.replace('-', '')
            encoding = encoding.replace('_', '')
            locale = lang + '.' + encoding
        data[locale] = alias
    return data

def parse_glibc_supported(filename):
    if False:
        return 10
    with open(filename, encoding='latin1') as f:
        lines = list(f)
    data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[:1] == '#':
            continue
        line = line.replace('/', ' ').strip()
        line = line.rstrip('\\').rstrip()
        words = line.split()
        if len(words) != 2:
            continue
        (alias, alias_encoding) = words
        locale = alias.lower()
        if '.' in locale:
            (lang, encoding) = locale.split('.')[:2]
            encoding = encoding.replace('-', '')
            encoding = encoding.replace('_', '')
            locale = lang + '.' + encoding
        (alias, _, modifier) = alias.partition('@')
        alias = _locale._replace_encoding(alias, alias_encoding)
        if modifier and (not (modifier == 'euro' and alias_encoding == 'ISO-8859-15')):
            alias += '@' + modifier
        data[locale] = alias
    return data

def pprint(data):
    if False:
        return 10
    items = sorted(data.items())
    for (k, v) in items:
        print('    %-40s%a,' % ('%a:' % k, v))

def print_differences(data, olddata):
    if False:
        i = 10
        return i + 15
    items = sorted(olddata.items())
    for (k, v) in items:
        if k not in data:
            print('#    removed %a' % k)
        elif olddata[k] != data[k]:
            print('#    updated %a -> %a to %a' % (k, olddata[k], data[k]))

def optimize(data):
    if False:
        print('Hello World!')
    locale_alias = locale.locale_alias
    locale.locale_alias = data.copy()
    for (k, v) in data.items():
        del locale.locale_alias[k]
        if locale.normalize(k) != v:
            locale.locale_alias[k] = v
    newdata = locale.locale_alias
    errors = check(data)
    locale.locale_alias = locale_alias
    if errors:
        sys.exit(1)
    return newdata

def check(data):
    if False:
        return 10
    errors = 0
    for (k, v) in data.items():
        if locale.normalize(k) != v:
            print('ERROR: %a -> %a != %a' % (k, locale.normalize(k), v), file=sys.stderr)
            errors += 1
    return errors
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--locale-alias', default=LOCALE_ALIAS, help='location of the X11 alias file (default: %a)' % LOCALE_ALIAS)
    parser.add_argument('--glibc-supported', default=SUPPORTED, help='location of the glibc SUPPORTED locales file (default: %a)' % SUPPORTED)
    args = parser.parse_args()
    data = locale.locale_alias.copy()
    data.update(parse_glibc_supported(args.glibc_supported))
    data.update(parse(args.locale_alias))
    while True:
        n = len(data)
        data = optimize(data)
        if len(data) == n:
            break
    print_differences(data, locale.locale_alias)
    print()
    print('locale_alias = {')
    pprint(data)
    print('}')