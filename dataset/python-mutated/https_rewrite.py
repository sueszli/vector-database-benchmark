"""
searx is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

searx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with searx. If not, see < http://www.gnu.org/licenses/ >.

(C) 2013- by Adam Tauber, <asciimoo@gmail.com>
"""
import re
from urllib.parse import urlparse
from lxml import etree
from os import listdir, environ
from os.path import isfile, isdir, join
from searx.plugins import logger
from flask_babel import gettext
from searx import searx_dir
name = 'HTTPS rewrite'
description = gettext('Rewrite HTTP links to HTTPS if possible')
default_on = True
preference_section = 'privacy'
if 'SEARX_HTTPS_REWRITE_PATH' in environ:
    rules_path = environ['SEARX_rules_path']
else:
    rules_path = join(searx_dir, 'plugins/https_rules')
logger = logger.getChild('https_rewrite')
https_rules = []

def load_single_https_ruleset(rules_path):
    if False:
        while True:
            i = 10
    ruleset = ()
    parser = etree.XMLParser()
    try:
        tree = etree.parse(rules_path, parser)
    except:
        return ()
    root = tree.getroot()
    if root.tag != 'ruleset':
        return ()
    if root.attrib.get('default_off'):
        return ()
    if root.attrib.get('platform'):
        return ()
    hosts = []
    rules = []
    exclusions = []
    for ruleset in root:
        if ruleset.tag == 'target':
            if not ruleset.attrib.get('host'):
                continue
            host = ruleset.attrib.get('host').replace('.', '\\.').replace('*', '.*')
            hosts.append(host)
        elif ruleset.tag == 'rule':
            if not ruleset.attrib.get('from') or not ruleset.attrib.get('to'):
                continue
            rule_from = ruleset.attrib['from'].replace('$', '\\')
            if rule_from.endswith('\\'):
                rule_from = rule_from[:-1] + '$'
            rule_to = ruleset.attrib['to'].replace('$', '\\')
            if rule_to.endswith('\\'):
                rule_to = rule_to[:-1] + '$'
            try:
                rules.append((re.compile(rule_from, re.I | re.U), rule_to))
            except:
                continue
        elif ruleset.tag == 'exclusion':
            if not ruleset.attrib.get('pattern'):
                continue
            exclusion_rgx = re.compile(ruleset.attrib.get('pattern'))
            exclusions.append(exclusion_rgx)
    try:
        target_hosts = re.compile('^(' + '|'.join(hosts) + ')', re.I | re.U)
    except:
        return ()
    return (target_hosts, rules, exclusions)

def load_https_rules(rules_path):
    if False:
        for i in range(10):
            print('nop')
    if not isdir(rules_path):
        logger.error("directory not found: '" + rules_path + "'")
        return
    xml_files = [join(rules_path, f) for f in listdir(rules_path) if isfile(join(rules_path, f)) and f[-4:] == '.xml']
    for ruleset_file in xml_files:
        ruleset = load_single_https_ruleset(ruleset_file)
        if not ruleset:
            continue
        https_rules.append(ruleset)
    logger.info('{n} rules loaded'.format(n=len(https_rules)))

def https_url_rewrite(result):
    if False:
        print('Hello World!')
    skip_https_rewrite = False
    for (target, rules, exclusions) in https_rules:
        if target.match(result['parsed_url'].netloc):
            for exclusion in exclusions:
                if exclusion.match(result['url']):
                    skip_https_rewrite = True
                    break
            if skip_https_rewrite:
                break
            for rule in rules:
                try:
                    new_result_url = rule[0].sub(rule[1], result['url'])
                except:
                    break
                new_parsed_url = urlparse(new_result_url)
                if result['url'] == new_result_url:
                    continue
                old_result_domainname = '.'.join(result['parsed_url'].hostname.split('.')[-2:])
                new_result_domainname = '.'.join(new_parsed_url.hostname.split('.')[-2:])
                if old_result_domainname == new_result_domainname:
                    result['url'] = new_result_url
            break
    return result

def on_result(request, search, result):
    if False:
        i = 10
        return i + 15
    if 'parsed_url' not in result:
        return True
    if result['parsed_url'].scheme == 'http':
        https_url_rewrite(result)
    return True
load_https_rules(rules_path)