import os
import sys
import re
import argparse
import tldextract
import itertools

def unique(tab):
    if False:
        for i in range(10):
            print('nop')
    final = []
    for i in tab:
        i = i.strip()
        if len(i):
            if not i in final:
                final.append(i)
    return final

def is_int(str):
    if False:
        i = 10
        return i + 15
    a = 'abcdefghijklmnopqrstuvwxyz'
    for c in a:
        if c in str:
            return False
    return True

def explode(str):
    if False:
        print('Hello World!')
    tab = []
    match = re.findall('[a-zA-Z0-9]+', str)
    for w in match:
        if is_int(w):
            min = int(w) - 10
            if min < 0:
                min = 0
            max = int(w) + 11
            for n in range(min, max):
                tab.append('%d' % n)
                tab.append('%01d' % n)
                tab.append('%02d' % n)
                tab.append('%03d' % n)
        else:
            tab.append(w)
    return tab
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subdomains', help='subdomains file (requires)')
parser.add_argument('-w', '--wordlist', help='wordlists for alteration (required)')
parser.parse_args()
args = parser.parse_args()
subdomains = []
if args.subdomains:
    if os.path.isfile(args.subdomains):
        fp = open(args.subdomains)
        subdomains = fp.read().split('\n')
        fp.close()
    else:
        parser.error('no subdomains found')
else:
    parser.error('no subdomains found')
wordlist = []
if args.wordlist:
    if os.path.isfile(args.wordlist):
        fp = open(args.wordlist)
        wordlist = fp.read().split('\n')
        fp.close()
    else:
        parser.error('no wordlist found')
else:
    parser.error('no wordlist found')
done = []
wordlist_sudomains = []
new_subdomains = []
for sub in subdomains:
    sub = sub.strip()
    if not len(sub):
        continue
    t_parse = tldextract.extract(sub)
    new_subdomain = t_parse.domain + '.' + t_parse.domain + '.' + t_parse.suffix
    if not new_subdomain in new_subdomains:
        new_subdomains.append(new_subdomain)
    if not t_parse.subdomain in done:
        done.append(t_parse.subdomain)
        new_words = explode(t_parse.subdomain)
        if type(new_words) is list and len(new_words):
            wordlist_sudomains = wordlist_sudomains + new_words
    if not t_parse.domain in done:
        done.append(t_parse.domain)
        new_words = explode(t_parse.domain)
        if type(new_words) is list and len(new_words):
            wordlist_sudomains = wordlist_sudomains + new_words
subdomains = subdomains + new_subdomains
wordlist_sudomains = unique(wordlist_sudomains)
wordlist = wordlist + wordlist_sudomains
wordlist = unique(wordlist)

def create_alts(sub, wordlist):
    if False:
        while True:
            i = 10
    t_parse = tldextract.extract(sub)
    subdomain_words = re.findall('[a-zA-Z0-9]+', t_parse.subdomain)
    for w in wordlist:
        t_words = subdomain_words + [w]
        to_glue = []
        for i in range(1, len(t_words) + 1):
            to_glue = to_glue + list(itertools.product(t_words, repeat=i))
        gluagisation(to_glue, t_parse.domain + '.' + t_parse.suffix)

def gluagisation(words_perms, domain):
    if False:
        while True:
            i = 10
    for one_perm in words_perms:
        gluagisation_single(one_perm, domain)

def gluagisation_single(one_perm, domain):
    if False:
        i = 10
        return i + 15
    l = len(one_perm)
    ll = l - 1
    if l == 1:
        new_sub = one_perm[0]
        new_sub = new_sub + '.' + domain
    else:
        for glue in t_glue_perms[ll]:
            j = 0
            k = 0
            new_sub = one_perm[0]
            for i in range(1, l):
                new_sub = new_sub + glue[j] + one_perm[i]
                j = j + 1
            new_sub = new_sub + '.' + domain
            if not new_sub in t_final:
                print(new_sub)
                t_final.append(new_sub)
glues = ['', '.', '-']
t_final = []
t_glue_perms = {0: [['']]}
for i in range(1, 10):
    t_glue_perms[i] = list(itertools.product(glues, repeat=i))
for sub in subdomains:
    sub = sub.strip()
    if not len(sub):
        continue
    create_alts(sub, wordlist)
for f in t_final:
    print(f)
exit()