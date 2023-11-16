"""
Reference: https://github.com/ProjectAnte/dnsgen
"""
import re
import time
import itertools
from config import settings
from modules import wildcard
from common import utils
from common import resolve
from common import request
from common.domain import Domain
from common.module import Module
from config.log import logger

def split_domain(domain):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split domain base on subdomain levels\n    Root+TLD is taken as one part, regardless of its levels\n    '
    ext = Domain(domain).extract()
    subname = ext.subdomain
    parts = ext.subdomain.split('.') + [ext.registered_domain]
    return (subname, parts)

class Altdns(Module):

    def __init__(self, domain):
        if False:
            i = 10
            return i + 15
        Module.__init__(self)
        self.module = 'Altdns'
        self.source = 'Altdns'
        self.start = time.time()
        self.domain = domain
        self.words = set()
        self.now_subdomains = set()
        self.new_subdomains = set()
        self.wordlen = 6
        self.num_count = 3

    def get_words(self):
        if False:
            return 10
        path = settings.data_storage_dir.joinpath('altdns_wordlist.txt')
        with open(path) as fd:
            for line in fd:
                word = line.lower().strip()
                if word:
                    self.words.add(word)

    def extract_words(self):
        if False:
            while True:
                i = 10
        "\n        Extend the dictionary based on target's domain naming conventions\n        "
        for subdomain in self.now_subdomains:
            (_, parts) = split_domain(subdomain)
            tokens = set(itertools.chain(*[word.lower().split('-') for word in parts]))
            tokens = tokens.union({word.lower() for word in parts})
            for token in tokens:
                if len(token) >= self.wordlen:
                    self.words.add(token)

    def increase_num(self, subname):
        if False:
            while True:
                i = 10
        '\n        If number is found in existing subdomain,\n        increase this number without any other alteration.\n        '
        count = 0
        digits = re.findall('\\d{1,3}', subname)
        for d in digits:
            for m in range(self.num_count):
                replacement = str(int(d) + 1 + m).zfill(len(d))
                tmp_domain = subname.replace(d, replacement)
                new_domain = f'{tmp_domain}.{self.domain}'
                self.new_subdomains.add(new_domain)
                count += 1
        logger.log('DEBUG', f'The increase_num generated {count} subdomains')

    def decrease_num(self, subname):
        if False:
            for i in range(10):
                print('nop')
        '\n        If number is found in existing subdomain,\n        decrease this number without any other alteration.\n        '
        count = 0
        digits = re.findall('\\d{1,3}', subname)
        for d in digits:
            for m in range(self.num_count):
                new_digit = int(d) - 1 - m
                if new_digit < 0:
                    break
                replacement = str(new_digit).zfill(len(d))
                tmp_domain = subname.replace(d, replacement)
                new_domain = f'{tmp_domain}.{self.domain}'
                self.new_subdomains.add(new_domain)
                count += 1
        logger.log('DEBUG', f'The decrease_num generated {count} subdomains')

    def insert_word(self, parts):
        if False:
            return 10
        '\n        Create new subdomain levels by inserting the words between existing levels\n        '
        count = 0
        for word in self.words:
            for index in range(len(parts)):
                tmp_parts = parts.copy()
                tmp_parts.insert(index, word)
                new_domain = '.'.join(tmp_parts)
                self.new_subdomains.add(new_domain)
                count += 1
        logger.log('DEBUG', f'The insert_word generated {count} subdomains')

    def add_word(self, subnames):
        if False:
            while True:
                i = 10
        '\n        On every subdomain level, prepend existing content with WORD-`,\n        append existing content with `-WORD`\n        '
        count = 0
        for word in self.words:
            for (index, name) in enumerate(subnames):
                tmp_subnames = subnames.copy()
                tmp_subnames[index] = f'{word}-{name}'
                new_subname = '.'.join(tmp_subnames + [self.domain])
                self.new_subdomains.add(new_subname)
                tmp_subnames = subnames.copy()
                tmp_subnames[index] = f'{name}-{word}'
                new_subname = '.'.join(tmp_subnames + [self.domain])
                self.new_subdomains.add(new_subname)
                count += 1
        logger.log('DEBUG', f'The add_word generated {count} subdomains')

    def replace_word(self, subname):
        if False:
            i = 10
            return i + 15
        '\n        If word longer than 3 is found in existing subdomain,\n        replace it with other words from the dictionary\n        '
        count = 0
        for word in self.words:
            if word not in subname:
                continue
            for word_alt in self.words:
                if word == word_alt:
                    continue
                new_subname = subname.replace(word, word_alt)
                new_subdomain = f'{new_subname}.{self.domain}'
                self.new_subdomains.add(new_subdomain)
                count += 1
        logger.log('DEBUG', f'The replace_word generated {count} subdomains')

    def gen_new_subdomains(self):
        if False:
            i = 10
            return i + 15
        for subdomain in self.now_subdomains:
            (subname, parts) = split_domain(subdomain)
            subnames = subname.split('.')
            if settings.altdns_increase_num:
                self.increase_num(subname)
            if settings.altdns_decrease_num:
                self.decrease_num(subname)
            if settings.altdns_replace_word:
                self.replace_word(subname)
            if settings.altdns_insert_word:
                self.insert_word(parts)
            if settings.altdns_add_word:
                self.add_word(subnames)
        count = len(self.new_subdomains)
        logger.log('DEBUG', f'The altdns module generated {count} subdomains')

    def run(self, data, port):
        if False:
            for i in range(10):
                print('nop')
        logger.log('INFOR', f'Start altdns module')
        self.now_subdomains = utils.get_subdomains(data)
        self.get_words()
        self.extract_words()
        self.gen_new_subdomains()
        self.subdomains = self.new_subdomains - self.now_subdomains
        count = len(self.subdomains)
        logger.log('INFOR', f'The altdns module generated {count} new subdomains')
        self.end = time.time()
        self.elapse = round(self.end - self.start, 1)
        self.gen_result()
        resolved_data = resolve.run_resolve(self.domain, self.results)
        valid_data = wildcard.deal_wildcard(resolved_data)
        request.run_request(self.domain, valid_data, port)