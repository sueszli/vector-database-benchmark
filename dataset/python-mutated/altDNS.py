import os
import re
import tld
import subprocess
import shlex
from collections import Counter
from app.config import Config
from app import utils
from .massdns import MassDNS
logger = utils.get_logger()
NUM_COUNT = 4

class DnsGen(object):

    def __init__(self, subdomains, words, base_domain=None):
        if False:
            return 10
        self.subdomains = subdomains
        self.base_domain = base_domain
        self.words = words

    def partiate_domain(self, domain):
        if False:
            i = 10
            return i + 15
        '\n        Split domain base on subdomain levels.\n        TLD is taken as one part, regardless of its levels (.co.uk, .com, ...)\n        '
        if self.base_domain:
            subdomain = re.sub(re.escape('.' + self.base_domain) + '$', '', domain)
            return subdomain.split('.') + [self.base_domain]
        ext = tld.get_tld(domain.lower(), fail_silently=True, as_object=True, fix_protocol=True)
        base_domain = '{}.{}'.format(ext.domain, ext.suffix)
        parts = ext.subdomain.split('.') + [base_domain]
        return [p for p in parts if p]

    def insert_word_every_index(self, parts):
        if False:
            return 10
        '\n        Create new subdomain levels by inserting the words between existing levels\n        '
        domains = []
        for w in self.words:
            for i in range(len(parts)):
                if i + 1 == len(parts):
                    break
                if w in parts[:-1]:
                    continue
                tmp_parts = parts[:-1]
                tmp_parts.insert(i, w)
                domains.append('{}.{}'.format('.'.join(tmp_parts), parts[-1]))
        return domains

    def insert_num_every_index(self, parts):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create new subdomain levels by inserting the numbers between existing levels\n        '
        domains = []
        for num in range(NUM_COUNT):
            for i in range(len(parts[:-1])):
                if num == 0:
                    continue
                tmp_parts = parts[:-1]
                tmp_parts[i] = '{}{}'.format(tmp_parts[i], num)
                domains.append('{}.{}'.format('.'.join(tmp_parts), '.'.join(parts[-1:])))
        return domains

    def prepend_word_every_index(self, parts):
        if False:
            print('Hello World!')
        '\n        On every subdomain level, prepend existing content with `WORD` and `WORD-`\n        '
        domains = []
        for w in self.words:
            for i in range(len(parts[:-1])):
                if w in parts[:-1]:
                    continue
                tmp_parts = parts[:-1]
                tmp_parts[i] = '{}{}'.format(w, tmp_parts[i])
                domains.append('{}.{}'.format('.'.join(tmp_parts), parts[-1]))
                tmp_parts = parts[:-1]
                tmp_parts[i] = '{}-{}'.format(w, tmp_parts[i])
                domains.append('{}.{}'.format('.'.join(tmp_parts), parts[-1]))
        return domains

    def append_word_every_index(self, parts):
        if False:
            return 10
        '\n        On every subdomain level, append existing content with `WORD` and `WORD-`\n        '
        domains = []
        for w in self.words:
            for i in range(len(parts[:-1])):
                if w in parts[:-1]:
                    continue
                tmp_parts = parts[:-1]
                tmp_parts[i] = '{}{}'.format(tmp_parts[i], w)
                domains.append('{}.{}'.format('.'.join(tmp_parts), '.'.join(parts[-1:])))
                tmp_parts = parts[:-1]
                tmp_parts[i] = '{}-{}'.format(tmp_parts[i], w)
                domains.append('{}.{}'.format('.'.join(tmp_parts), '.'.join(parts[-1:])))
        return domains

    def replace_word_with_word(self, parts):
        if False:
            for i in range(10):
                print('nop')
        '\n        If word longer than 3 is found in existing subdomain, replace it with other words from the dictionary\n        '
        domains = []
        for w in self.words:
            if len(w) <= 3:
                continue
            if w in '.'.join(parts[:-1]):
                for w_alt in self.words:
                    if w == w_alt:
                        continue
                    if w in parts[:-1]:
                        continue
                    domains.append('{}.{}'.format('.'.join(parts[:-1]).replace(w, w_alt), '.'.join(parts[-1:])))
        return domains

    def run(self):
        if False:
            return 10
        for domain in set(self.subdomains):
            parts = self.partiate_domain(domain)
            permutations = []
            permutations += self.insert_word_every_index(parts)
            permutations += self.insert_num_every_index(parts)
            permutations += self.prepend_word_every_index(parts)
            permutations += self.append_word_every_index(parts)
            permutations += self.replace_word_with_word(parts)
            for perm in permutations:
                yield perm

class AltDNS(object):

    def __init__(self, subdomains, base_domain, words, wildcard_domain_ip=None):
        if False:
            print('Hello World!')
        self.subdomains = subdomains
        self.base_domain = base_domain
        self.words = words
        if wildcard_domain_ip is None:
            wildcard_domain_ip = []
        self.wildcard_domain_ip = wildcard_domain_ip

    def run(self):
        if False:
            i = 10
            return i + 15
        domains = DnsGen(set(self.subdomains), self.words, base_domain=self.base_domain).run()
        logger.info('start AltDNS:{} wildcard_record:{}'.format(self.base_domain, ','.join(self.wildcard_domain_ip)))
        mass = MassDNS(domains, mass_dns_bin=Config.MASSDNS_BIN, dns_server=Config.DNS_SERVER, tmp_dir=Config.TMP_PATH, wildcard_domain_ip=self.wildcard_domain_ip, concurrent=Config.ALT_DNS_CONCURRENT)
        return mass.run()
"\n[{\n\t'domain': 'account.tophant.com',\n\t'type': 'A',\n\t'record': '182.254.150.199'\n}]\n"

def alt_dns(subdomains, base_domain=None, words=None, wildcard_domain_ip=None):
    if False:
        i = 10
        return i + 15
    if len(subdomains) == 0:
        return []
    a = AltDNS(subdomains, base_domain, words=words, wildcard_domain_ip=wildcard_domain_ip)
    raw_domains_info = a.run()
    '解决泛解析的问题'
    domains_info = []
    records = [x['record'] for x in raw_domains_info]
    records_count = Counter(records)
    for info in raw_domains_info:
        if records_count[info['record']] >= 15:
            continue
        domains_info.append(info)
    return domains_info