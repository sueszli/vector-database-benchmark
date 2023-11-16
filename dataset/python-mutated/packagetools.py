import re
from urllib.parse import urlparse

def match_first(string, *args):
    if False:
        while True:
            i = 10
    '\n    matches against list of regexp and returns first match.\n    '
    for patternlist in args:
        for pattern in patternlist:
            r = pattern.search(string)
            if r is not None:
                name = r.group(1)
                return name
    return string

def parse_names(files):
    if False:
        return 10
    '\n    Generates packages names from name, data lists.\n\n    :param files: list of (name, data)\n    :return: packagenames mapt to data lists (eg. urls)\n    '
    packs = {}
    endings = '\\.(3gp|7zip|7z|abr|ac3|aiff|aifc|aif|ai|au|avi|bin|bz2|cbr|cbz|ccf|cue|cvd|chm|dta|deb|divx|djvu|dlc|dmg|doc|docx|dot|eps|exe|ff|flv|f4v|gsd|gif|gz|iwd|iso|ipsw|java|jar|jpg|jpeg|jdeatme|load|mws|mw|m4v|m4a|mkv|mp2|mp3|mp4|mov|movie|mpeg|mpe|mpg|msi|msu|msp|nfo|npk|oga|ogg|ogv|otrkey|pkg|png|pdf|pptx|ppt|pps|ppz|pot|psd|qt|rmvb|rm|rar|ram|ra|rev|rnd|r\\d+|rpm|run|rsdf|rtf|sh(!?tml)|srt|snd|sfv|swf|tar|tif|tiff|ts|txt|viv|vivo|vob|wav|wmv|xla|xls|xpi|zeno|zip|z\\d+|_[_a-z]{2}|\\d+$)'
    rar_pats = [re.compile('(.*)(\\.|_|-)pa?r?t?\\.?[0-9]+.(rar|exe)$', re.I), re.compile('(.*)(\\.|_|-)part\\.?[0]*[1].(rar|exe)$', re.I), re.compile('(.*)\\.rar$', re.I), re.compile('(.*)\\.r\\d+$', re.I), re.compile('(.*)(\\.|_|-)\\d+$', re.I)]
    zip_pats = [re.compile('(.*)\\.zip$', re.I), re.compile('(.*)\\.z\\d+$', re.I), re.compile('(?is).*\\.7z\\.[\\d]+$', re.I), re.compile('(.*)\\.a.$', re.I)]
    ffsj_pats = [re.compile('(.*)\\._((_[a-z])|([a-z]{2}))(\\.|$)'), re.compile('(.*)(\\.|_|-)[\\d]+(' + endings + '$)', re.I)]
    isz_pats = [re.compile('(.*)\\.isz$', re.I), re.compile('(.*)\\.i\\d{2}$', re.I)]
    pat1 = re.compile('(\\.?CD\\d+)', re.I)
    pat2 = re.compile('(\\.?part\\d+)', re.I)
    pat3 = re.compile('(.+)[\\.\\-_]+$')
    pat4 = re.compile('(.+)\\.\\d+\\.xtm$')
    for (file, url) in files:
        pattern_match = False
        if file is None:
            continue
        name = file.rstrip('/')
        split = name.rsplit('/', 1)
        if len(split) > 1:
            name = split.pop(1)
        before = name
        name = match_first(name, rar_pats, zip_pats, isz_pats, ffsj_pats)
        if before != name:
            pattern_match = True
        r = pat4.search(name)
        if r is not None:
            name = r.group(1)
        r = pat1.search(name)
        if r is not None:
            name = name.replace(r.group(0), '')
            pattern_match = True
        r = pat2.search(name)
        if r is not None:
            name = name.replace(r.group(0), '')
            pattern_match = True
        if pattern_match:
            index = name.rfind('.')
            if index <= 0:
                index = name.rfind('_')
            if index > 0:
                length = len(name) - index
                if length <= 4:
                    name = name[:-length]
            r = pat3.search(name)
            if r is not None:
                name = r.group(1)
            name = name.replace('.', ' ')
            name = name.replace('_', ' ')
            name = name.strip()
        else:
            name = ''
        if not name:
            name = urlparse(file).hostname
            if name:
                name = name.replace('www.', '')
        if not name:
            name = 'unknown'
        if name in packs:
            packs[name].append(url)
        else:
            packs[name] = [url]
    return packs