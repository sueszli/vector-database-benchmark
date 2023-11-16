"""Functions to load skill data such as intents and regular expressions."""
import collections
import csv
import re
from os import walk
from os.path import splitext, join
from mycroft.util.format import expand_options
from mycroft.util.log import LOG

def read_vocab_file(path):
    if False:
        for i in range(10):
            print('nop')
    ' Read voc file.\n\n        This reads a .voc file, stripping out empty lines comments and expand\n        parentheses. It returns each line as a list of all expanded\n        alternatives.\n\n        Args:\n            path (str): path to vocab file.\n\n        Returns:\n            List of Lists of strings.\n    '
    vocab = []
    with open(path, 'r', encoding='utf8') as voc_file:
        for line in voc_file.readlines():
            if line.startswith('#') or line.strip() == '':
                continue
            vocab.append(expand_options(line.lower()))
    return vocab

def load_regex_from_file(path, skill_id):
    if False:
        while True:
            i = 10
    'Load regex from file\n    The regex is sent to the intent handler using the message bus\n\n    Args:\n        path:       path to vocabulary file (*.voc)\n        skill_id:   skill_id to the regex is tied to\n    '
    regexes = []
    if path.endswith('.rx'):
        with open(path, 'r', encoding='utf8') as reg_file:
            for line in reg_file.readlines():
                if line.startswith('#'):
                    continue
                LOG.debug('regex pre-munge: ' + line.strip())
                regex = munge_regex(line.strip(), skill_id)
                LOG.debug('regex post-munge: ' + regex)
                re.compile(regex)
                regexes.append(regex)
    return regexes

def load_vocabulary(basedir, skill_id):
    if False:
        for i in range(10):
            print('nop')
    'Load vocabulary from all files in the specified directory.\n\n    Args:\n        basedir (str): path of directory to load from (will recurse)\n        skill_id: skill the data belongs to\n    Returns:\n        dict with intent_type as keys and list of list of lists as value.\n    '
    vocabs = {}
    for (path, _, files) in walk(basedir):
        for f in files:
            if f.endswith('.voc'):
                vocab_type = to_alnum(skill_id) + splitext(f)[0]
                vocs = read_vocab_file(join(path, f))
                if vocs:
                    vocabs[vocab_type] = vocs
    return vocabs

def load_regex(basedir, skill_id):
    if False:
        print('Hello World!')
    'Load regex from all files in the specified directory.\n\n    Args:\n        basedir (str): path of directory to load from\n        bus (messagebus emitter): messagebus instance used to send the vocab to\n                                  the intent service\n        skill_id (str): skill identifier\n    '
    regexes = []
    for (path, _, files) in walk(basedir):
        for f in files:
            if f.endswith('.rx'):
                regexes += load_regex_from_file(join(path, f), skill_id)
    return regexes

def to_alnum(skill_id):
    if False:
        while True:
            i = 10
    'Convert a skill id to only alphanumeric characters\n\n     Non alpha-numeric characters are converted to "_"\n\n    Args:\n        skill_id (str): identifier to be converted\n    Returns:\n        (str) String of letters\n    '
    return ''.join((c if c.isalnum() else '_' for c in str(skill_id)))

def munge_regex(regex, skill_id):
    if False:
        while True:
            i = 10
    'Insert skill id as letters into match groups.\n\n    Args:\n        regex (str): regex string\n        skill_id (str): skill identifier\n    Returns:\n        (str) munged regex\n    '
    base = '(?P<' + to_alnum(skill_id)
    return base.join(regex.split('(?P<'))

def munge_intent_parser(intent_parser, name, skill_id):
    if False:
        for i in range(10):
            print('nop')
    "Rename intent keywords to make them skill exclusive\n    This gives the intent parser an exclusive name in the\n    format <skill_id>:<name>.  The keywords are given unique\n    names in the format <Skill id as letters><Intent name>.\n\n    The function will not munge instances that's already been\n    munged\n\n    Args:\n        intent_parser: (IntentParser) object to update\n        name: (str) Skill name\n        skill_id: (int) skill identifier\n    "
    if not name.startswith(str(skill_id) + ':'):
        intent_parser.name = str(skill_id) + ':' + name
    else:
        intent_parser.name = name
    skill_id = to_alnum(skill_id)
    reqs = []
    for i in intent_parser.requires:
        if not i[0].startswith(skill_id):
            kw = (skill_id + i[0], skill_id + i[0])
            reqs.append(kw)
        else:
            reqs.append(i)
    intent_parser.requires = reqs
    opts = []
    for i in intent_parser.optional:
        if not i[0].startswith(skill_id):
            kw = (skill_id + i[0], skill_id + i[0])
            opts.append(kw)
        else:
            opts.append(i)
    intent_parser.optional = opts
    at_least_one = []
    for i in intent_parser.at_least_one:
        element = [skill_id + e.replace(skill_id, '') for e in i]
        at_least_one.append(tuple(element))
    intent_parser.at_least_one = at_least_one

def read_value_file(filename, delim):
    if False:
        i = 10
        return i + 15
    'Read value file.\n\n    The value file is a simple csv structure with a key and value.\n\n    Args:\n        filename (str): file to read\n        delim (str): csv delimiter\n\n    Returns:\n        OrderedDict with results.\n    '
    result = collections.OrderedDict()
    if filename:
        with open(filename) as f:
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                if len(row) != 2:
                    continue
                result[row[0]] = row[1]
    return result

def read_translated_file(filename, data):
    if False:
        while True:
            i = 10
    'Read a file inserting data.\n\n    Args:\n        filename (str): file to read\n        data (dict): dictionary with data to insert into file\n\n    Returns:\n        list of lines.\n    '
    if filename:
        with open(filename) as f:
            text = f.read().replace('{{', '{').replace('}}', '}')
            return text.format(**data or {}).rstrip('\n').split('\n')
    else:
        return None