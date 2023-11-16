import re
import requests
import random
from core.utils import verb, xml_parser
from core.colors import run, good
from plugins.wayback import time_machine

def zap(input_url, archive, domain, host, internal, robots, proxies):
    if False:
        return 10
    'Extract links from robots.txt and sitemap.xml.'
    if archive:
        print('%s Fetching URLs from archive.org' % run)
        if False:
            archived_urls = time_machine(domain, 'domain')
        else:
            archived_urls = time_machine(host, 'host')
        print('%s Retrieved %i URLs from archive.org' % (good, len(archived_urls) - 1))
        for url in archived_urls:
            verb('Internal page', url)
            internal.add(url)
    response = requests.get(input_url + '/robots.txt', proxies=random.choice(proxies)).text
    if '<body' not in response:
        matches = re.findall('Allow: (.*)|Disallow: (.*)', response)
        if matches:
            for match in matches:
                match = ''.join(match)
                if '*' not in match:
                    url = input_url + match
                    internal.add(url)
                    robots.add(url)
            print('%s URLs retrieved from robots.txt: %s' % (good, len(robots)))
    response = requests.get(input_url + '/sitemap.xml', proxies=random.choice(proxies)).text
    if '<body' not in response:
        matches = xml_parser(response)
        if matches:
            print('%s URLs retrieved from sitemap.xml: %s' % (good, len(matches)))
            for match in matches:
                verb('Internal page', match)
                internal.add(match)