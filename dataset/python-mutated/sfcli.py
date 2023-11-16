import argparse
import cmd
import codecs
import io
import json
import os
import re
import shlex
import sys
import time
from os.path import expanduser
import requests
ASCII_LOGO = '\n  _________      .__    .___          ___________            __\n /   _____/_____ |__| __| _/__________\\_   _____/___   _____/  |_\n \\_____  \\\\____ \\|  |/ __ |/ __ \\_  __ \\    __)/  _ \\ /  _ \\   __\\\n /        \\  |_> >  / /_/ \\  ___/|  | \\/     \\(  <_> |  <_> )  |\n/_______  /   __/|__\\____ |\\___  >__|  \\___  / \\____/ \\____/|__|\n        \\/|__|           \\/    \\/          \\/\n                Open Source Intelligence Automation.'
COPYRIGHT_INFO = '               by Steve Micallef | @spiderfoot\n'
try:
    import readline
except ImportError:
    import pyreadline as readline

class bcolors:
    GREYBLUE = '\x1b[38;5;25m'
    GREY = '\x1b[38;5;243m'
    DARKRED = '\x1b[38;5;124m'
    DARKGREEN = '\x1b[38;5;30m'
    BOLD = '\x1b[1m'
    ENDC = '\x1b[0m'
    GREYBLUE_DARK = '\x1b[38;5;24m'

class SpiderFootCli(cmd.Cmd):
    version = '4.0.0'
    pipecmd = None
    output = None
    modules = []
    types = []
    correlationrules = []
    prompt = 'sf> '
    nohelp = "[!] Unknown command '%s'."
    knownscans = []
    ownopts = {'cli.debug': False, 'cli.silent': False, 'cli.color': True, 'cli.output': 'pretty', 'cli.history': True, 'cli.history_file': '', 'cli.spool': False, 'cli.spool_file': '', 'cli.ssl_verify': True, 'cli.username': '', 'cli.password': '', 'cli.server_baseurl': 'http://127.0.0.1:5001'}

    def default(self, line):
        if False:
            i = 10
            return i + 15
        if line.startswith('#'):
            return
        self.edprint('Unknown command')

    def complete_start(self, text, line, startidx, endidx):
        if False:
            while True:
                i = 10
        return self.complete_default(text, line, startidx, endidx)

    def complete_find(self, text, line, startidx, endidx):
        if False:
            print('Hello World!')
        return self.complete_default(text, line, startidx, endidx)

    def complete_data(self, text, line, startidx, endidx):
        if False:
            for i in range(10):
                print('nop')
        return self.complete_default(text, line, startidx, endidx)

    def complete_default(self, text, line, startidx, endidx):
        if False:
            i = 10
            return i + 15
        ret = list()
        if not isinstance(text, str):
            return ret
        if not isinstance(line, str):
            return ret
        if '-m' in line and line.find('-m') > line.find('-t'):
            for m in self.modules:
                if m.startswith(text):
                    ret.append(m)
        if '-t' in line and line.find('-t') > line.find('-m'):
            for t in self.types:
                if t.startswith(text):
                    ret.append(t)
        return ret

    def dprint(self, msg, err=False, deb=False, plain=False, color=None):
        if False:
            print('Hello World!')
        cout = ''
        sout = ''
        pfx = ''
        col = ''
        if err:
            pfx = '[!]'
            if self.ownopts['cli.color']:
                col = bcolors.DARKRED
        else:
            pfx = '[*]'
            if self.ownopts['cli.color']:
                col = bcolors.DARKGREEN
        if deb:
            if not self.ownopts['cli.debug']:
                return
            pfx = '[+]'
            if self.ownopts['cli.color']:
                col = bcolors.GREY
        if color:
            pfx = ''
            col = color
        if err or not self.ownopts['cli.silent']:
            if not plain or color:
                cout = col + bcolors.BOLD + pfx + ' ' + bcolors.ENDC + col + msg + bcolors.ENDC
                sout = pfx + ' ' + msg
            else:
                cout = msg
                sout = msg
            print(cout)
        if self.ownopts['cli.spool']:
            f = codecs.open(self.ownopts['cli.spool_file'], 'a', encoding='utf-8')
            f.write(sout)
            f.write('\n')
            f.close()

    def do_debug(self, line):
        if False:
            return 10
        'debug\n        Short-cut command for set cli.debug = 1'
        if self.ownopts['cli.debug']:
            val = '0'
        else:
            val = '1'
        return self.do_set('cli.debug = ' + val)

    def do_spool(self, line):
        if False:
            for i in range(10):
                print('nop')
        'spool\n        Short-cut command for set cli.spool = 1/0'
        if self.ownopts['cli.spool']:
            val = '0'
        else:
            val = '1'
        if self.ownopts['cli.spool_file']:
            return self.do_set('cli.spool = ' + val)
        self.edprint("You haven't set cli.spool_file. Set that before enabling spooling.")
        return None

    def do_history(self, line):
        if False:
            for i in range(10):
                print('nop')
        'history [-l]\n        Short-cut command for set cli.history = 1/0.\n        Add -l to just list the history.'
        c = self.myparseline(line)
        if '-l' in c[0]:
            i = 0
            while i < readline.get_current_history_length():
                self.dprint(readline.get_history_item(i), plain=True)
                i += 1
            return None
        if self.ownopts['cli.history']:
            val = '0'
        else:
            val = '1'
        return self.do_set('cli.history = ' + val)

    def precmd(self, line):
        if False:
            for i in range(10):
                print('nop')
        if self.ownopts['cli.history'] and line != 'EOF':
            f = codecs.open(self.ownopts['cli.history_file'], 'a', encoding='utf-8')
            f.write(line)
            f.write('\n')
            f.close()
        if self.ownopts['cli.spool']:
            f = codecs.open(self.ownopts['cli.spool_file'], 'a', encoding='utf-8')
            f.write(self.prompt + line)
            f.write('\n')
            f.close()
        return line

    def ddprint(self, msg):
        if False:
            while True:
                i = 10
        self.dprint(msg, deb=True)

    def edprint(self, msg):
        if False:
            print('Hello World!')
        self.dprint(msg, err=True)

    def pretty(self, data, titlemap=None):
        if False:
            print('Hello World!')
        if not data:
            return ''
        out = list()
        maxsize = dict()
        if type(data[0]) == dict:
            cols = list(data[0].keys())
        else:
            cols = list(map(str, list(range(0, len(data[0])))))
        if titlemap:
            nc = list()
            for c in cols:
                if c in titlemap:
                    nc.append(c)
            cols = nc
        spaces = 2
        for r in data:
            for (i, c) in enumerate(r):
                if type(r) == list:
                    cn = str(i)
                    if type(c) == int:
                        v = str(c)
                    if type(c) == str:
                        v = c
                else:
                    cn = c
                    v = str(r[c])
                if len(v) > maxsize.get(cn, 0):
                    maxsize[cn] = len(v)
        if titlemap:
            for c in maxsize:
                if len(titlemap.get(c, c)) > maxsize[c]:
                    maxsize[c] = len(titlemap.get(c, c))
        for (i, c) in enumerate(cols):
            if titlemap:
                t = titlemap.get(c, c)
            else:
                t = c
            out.append(t)
            sdiff = maxsize[c] - len(t) + 1
            out.append(' ' * spaces)
            if sdiff > 0 and i < len(cols) - 1:
                out.append(' ' * sdiff)
        out.append('\n')
        for (i, c) in enumerate(cols):
            out.append('-' * (maxsize[c] + spaces))
            if i < len(cols) - 1:
                out.append('+')
        out.append('\n')
        for r in data:
            i = 0
            di = 0
            tr = type(r)
            for c in r:
                if tr == list:
                    cn = str(i)
                    tc = type(c)
                    if tc == int:
                        v = str(c)
                    if tc == str:
                        v = c
                else:
                    cn = c
                    v = str(r[c])
                if cn not in cols:
                    i += 1
                    continue
                out.append(v)
                lv = len(v)
                if di == 0:
                    sdiff = maxsize[cn] - lv + spaces
                else:
                    sdiff = maxsize[cn] - lv + spaces - 1
                if di < len(cols) - 1:
                    out.append(' ' * sdiff)
                if di < len(cols) - 1:
                    out.append('| ')
                di += 1
                i += 1
            out.append('\n')
        return ''.join(out)

    def request(self, url, post=None):
        if False:
            i = 10
            return i + 15
        if not url:
            self.edprint('Invalid request URL')
            return None
        if not isinstance(url, str):
            self.edprint(f'Invalid request URL: {url}')
            return None
        headers = {'User-agent': 'SpiderFoot-CLI/' + self.version, 'Accept': 'application/json'}
        try:
            self.ddprint(f'Fetching: {url}')
            if not post:
                r = requests.get(url, headers=headers, verify=self.ownopts['cli.ssl_verify'], auth=requests.auth.HTTPDigestAuth(self.ownopts['cli.username'], self.ownopts['cli.password']))
            else:
                self.ddprint(f'Posting: {post}')
                r = requests.post(url, headers=headers, verify=self.ownopts['cli.ssl_verify'], auth=requests.auth.HTTPDigestAuth(self.ownopts['cli.username'], self.ownopts['cli.password']), data=post)
            self.ddprint(f'Response: {r}')
            if r.status_code == requests.codes.ok:
                return r.text
            r.raise_for_status()
        except BaseException as e:
            self.edprint(f'Failed communicating with server: {e}')
            return None

    def emptyline(self):
        if False:
            return 10
        return

    def completedefault(self, text, line, begidx, endidx):
        if False:
            print('Hello World!')
        return []

    def myparseline(self, cmdline, replace=True):
        if False:
            for i in range(10):
                print('nop')
        ret = [list(), list()]
        if not cmdline:
            return ret
        try:
            s = shlex.split(cmdline)
        except Exception as e:
            self.edprint(f'Error parsing command: {e}')
            return ret
        for c in s:
            if c == '|':
                break
            if replace and c.startswith('$') and (c in self.ownopts):
                ret[0].append(self.ownopts[c])
            else:
                ret[0].append(c)
        if s.count('|') == 0:
            return ret
        ret[1] = list()
        i = 0
        ret[1].append(list())
        for t in s[s.index('|') + 1:]:
            if t == '|':
                i += 1
                ret[1].append(list())
            elif t.startswith('$') and t in self.ownopts:
                ret[1][i].append(self.ownopts[t])
            else:
                ret[1][i].append(t)
        return ret

    def send_output(self, data, cmd, titles=None, total=True, raw=False):
        if False:
            return 10
        out = None
        try:
            if raw:
                j = data
                totalrec = 0
            else:
                j = json.loads(data)
                totalrec = len(j)
        except BaseException as e:
            self.edprint(f'Unable to parse data from server: {e}')
            return
        if raw:
            out = data
        else:
            if self.ownopts['cli.output'] == 'json':
                out = json.dumps(j, indent=4, separators=(',', ': '))
            if self.ownopts['cli.output'] == 'pretty':
                out = self.pretty(j, titlemap=titles)
            if not out:
                self.edprint(f"Unknown output format '{self.ownopts['cli.output']}'.")
                return
        c = self.myparseline(cmd)
        if len(c[1]) == 0:
            self.dprint(out, plain=True)
            if total:
                self.dprint(f'Total records: {totalrec}')
            return
        for pc in c[1]:
            newout = ''
            if len(pc) == 0:
                self.edprint('Invalid syntax.')
                return
            pipecmd = pc[0]
            pipeargs = ' '.join(pc[1:])
            if pipecmd not in ['str', 'regex', 'file', 'grep', 'top', 'last']:
                self.edprint('Unrecognised pipe command.')
                return
            if pipecmd == 'regex':
                p = re.compile(pipeargs, re.IGNORECASE)
                for r in out.split('\n'):
                    if re.match(p, r.strip()):
                        newout += r + '\n'
            if pipecmd in ['str', 'grep']:
                for r in out.split('\n'):
                    if pipeargs.lower() in r.strip().lower():
                        newout += r + '\n'
            if pipecmd == 'top':
                if not pipeargs.isdigit():
                    self.edprint('Invalid syntax.')
                    return
                newout = '\n'.join(out.split('\n')[0:int(pipeargs)])
            if pipecmd == 'last':
                if not pipeargs.isdigit():
                    self.edprint('Invalid syntax.')
                    return
                tot = len(out.split('\n'))
                i = tot - int(pipeargs)
                newout = '\n'.join(out.split('\n')[i:])
            if pipecmd == 'file':
                try:
                    f = codecs.open(pipeargs, 'w', encoding='utf-8')
                    f.write(out)
                    f.close()
                except BaseException as e:
                    self.edprint(f'Unable to write to file: {e}')
                    return
                self.dprint(f"Successfully wrote to file '{pipeargs}'.")
                return
            out = newout
        self.dprint(newout, plain=True)

    def do_query(self, line):
        if False:
            print('Hello World!')
        'query <SQL query>\n        Run an <SQL query> against the database.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        query = ' '.join(c[0])
        d = self.request(self.ownopts['cli.server_baseurl'] + '/query', post={'query': query})
        if not d:
            return
        j = json.loads(d)
        if j[0] == 'ERROR':
            self.edprint(f'Error running your query: {j[1]}')
            return
        self.send_output(d, line)

    def do_ping(self, line):
        if False:
            while True:
                i = 10
        "ping\n        Ping the SpiderFoot server to ensure it's responding."
        d = self.request(self.ownopts['cli.server_baseurl'] + '/ping')
        if not d:
            return
        s = json.loads(d)
        if s[0] == 'SUCCESS':
            self.dprint(f"Server {self.ownopts['cli.server_baseurl']} responding.")
            self.do_modules('', cacheonly=True)
            self.do_types('', cacheonly=True)
        else:
            self.dprint(f'Something odd happened: {d}')
        if s[1] != self.version:
            self.edprint(f'Server and CLI version are not the same ({s[1]} / {self.version}). This could lead to unpredictable results!')

    def do_modules(self, line, cacheonly=False):
        if False:
            while True:
                i = 10
        'modules\n        List all available modules and their descriptions.'
        d = self.request(self.ownopts['cli.server_baseurl'] + '/modules')
        if not d:
            return
        if cacheonly:
            j = json.loads(d)
            for m in j:
                self.modules.append(m['name'])
            return
        self.send_output(d, line, titles={'name': 'Module name', 'descr': 'Description'})

    def do_correlationrules(self, line, cacheonly=False):
        if False:
            while True:
                i = 10
        'correlations\n        List all available correlation rules and their descriptions.'
        d = self.request(self.ownopts['cli.server_baseurl'] + '/correlationrules')
        if not d:
            return
        if cacheonly:
            j = json.loads(d)
            for m in j:
                self.correlationrules.append(m['name'])
            return
        self.send_output(d, line, titles={'id': 'Correlation rule ID', 'name': 'Name', 'risk': 'Risk'})

    def do_types(self, line, cacheonly=False):
        if False:
            for i in range(10):
                print('nop')
        'types\n        List all available element types and their descriptions.'
        d = self.request(self.ownopts['cli.server_baseurl'] + '/eventtypes')
        if not d:
            return
        if cacheonly:
            j = json.loads(d)
            for t in j:
                self.types.append(t[0])
            return
        self.send_output(d, line, titles={'1': 'Element description', '0': 'Element name'})

    def do_load(self, line):
        if False:
            print('Hello World!')
        'load <file>\n        Execute SpiderFoot CLI commands found in <file>.'
        pass

    def do_scaninfo(self, line):
        if False:
            while True:
                i = 10
        'scaninfo <sid> [-c]\n        Get status information for scan ID <sid>, optionally also its\n        configuration if -c is supplied.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        sid = c[0][0]
        d = self.request(self.ownopts['cli.server_baseurl'] + f'/scanopts?id={sid}')
        if not d:
            return
        j = json.loads(d)
        if len(j) == 0:
            self.dprint('No such scan exists.')
            return
        out = list()
        out.append(f"Name: {j['meta'][0]}")
        out.append(f'ID: {sid}')
        out.append(f"Target: {j['meta'][1]}")
        out.append(f"Started: {j['meta'][3]}")
        out.append(f"Completed: {j['meta'][4]}")
        out.append(f"Status: {j['meta'][5]}")
        if '-c' in c[0]:
            out.append('Configuration:')
            for k in sorted(j['config']):
                out.append(f"  {k} = {j['config'][k]}")
        self.send_output('\n'.join(out), line, total=False, raw=True)

    def do_scans(self, line):
        if False:
            for i in range(10):
                print('nop')
        'scans [-x]\n        List all scans, past and present. -x for extended view.'
        d = self.request(self.ownopts['cli.server_baseurl'] + '/scanlist')
        if not d:
            return
        j = json.loads(d)
        if len(j) == 0:
            self.dprint('No scans exist.')
            return
        c = self.myparseline(line)
        titles = dict()
        if '-x' in c[0]:
            titles = {'0': 'ID', '1': 'Name', '2': 'Target', '4': 'Started', '5': 'Finished', '6': 'Status', '7': 'Total Elements'}
        else:
            titles = {'0': 'ID', '2': 'Target', '6': 'Status', '7': 'Total Elements'}
        self.send_output(d, line, titles=titles)

    def do_correlations(self, line):
        if False:
            while True:
                i = 10
        'correlations <sid> [-c correlation_id]\n        Get the correlation results for scan ID <sid> and optionally the\n        events associated with a correlation result [correlation_id] to\n        get the results for a particular correlation.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        post = {'id': c[0][0]}
        if '-c' in c[0]:
            post['correlationId'] = c[0][c[0].index('-c') + 1]
            url = self.ownopts['cli.server_baseurl'] + '/scaneventresults'
            titles = {'10': 'Type', '1': 'Data'}
        else:
            url = self.ownopts['cli.server_baseurl'] + '/scancorrelations'
            titles = {'0': 'ID', '1': 'Title', '3': 'Risk', '7': 'Data Elements'}
        d = self.request(url, post=post)
        if not d:
            return
        j = json.loads(d)
        if len(j) < 1:
            self.dprint('No results.')
            return
        self.send_output(d, line, titles=titles)

    def do_data(self, line):
        if False:
            i = 10
            return i + 15
        'data <sid> [-t type] [-x] [-u]\n        Get the scan data for scan ID <sid> and optionally the element\n        type [type] (e.g. EMAILADDR), [type]. Use -x for extended format.\n        Use -u for a unique set of results.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        post = {'id': c[0][0]}
        if '-t' in c[0]:
            post['eventType'] = c[0][c[0].index('-t') + 1]
        else:
            post['eventType'] = 'ALL'
        if '-u' in c[0]:
            url = self.ownopts['cli.server_baseurl'] + '/scaneventresultsunique'
            titles = {'0': 'Data'}
        else:
            url = self.ownopts['cli.server_baseurl'] + '/scaneventresults'
            titles = {'10': 'Type', '1': 'Data'}
        d = self.request(url, post=post)
        if not d:
            return
        j = json.loads(d)
        if len(j) < 1:
            self.dprint('No results.')
            return
        if '-x' in c[0]:
            titles['0'] = 'Last Seen'
            titles['3'] = 'Module'
            titles['2'] = 'Source Data'
        d = d.replace('&lt;/SFURL&gt;', '').replace('&lt;SFURL&gt;', '')
        self.send_output(d, line, titles=titles)

    def do_export(self, line):
        if False:
            while True:
                i = 10
        'export <sid> [-t type] [-f file]\n        Export the scan data for scan ID <sid> as type [type] to file [file].\n        Valid types: csv, json, gexf (default: json).'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        export_format = 'json'
        if '-t' in c[0]:
            export_format = c[0][c[0].index('-t') + 1]
        file = None
        if '-f' in c[0]:
            file = c[0][c[0].index('-f') + 1]
        base_url = self.ownopts['cli.server_baseurl']
        post = {'ids': c[0][0]}
        if export_format not in ['json', 'csv', 'gexf']:
            self.edprint(f'Invalid export format: {export_format}')
            return
        data = None
        if export_format == 'json':
            res = self.request(base_url + '/scanexportjsonmulti', post=post)
            if not res:
                self.dprint('No results.')
                return
            j = json.loads(res)
            if len(j) < 1:
                self.dprint('No results.')
                return
            data = json.dumps(j)
        elif export_format == 'csv':
            data = self.request(base_url + '/scaneventresultexportmulti', post=post)
        elif export_format == 'gexf':
            data = self.request(base_url + '/scanvizmulti', post=post)
        if not data:
            self.dprint('No results.')
            return
        self.send_output(data, line, titles=None, total=False, raw=True)
        if file:
            try:
                with io.open(file, 'w', encoding='utf-8', errors='ignore') as fp:
                    fp.write(data)
                self.dprint(f'Wrote scan {c[0][0]} data to {file}')
            except Exception as e:
                self.edprint(f"Could not write scan {c[0][0]} data to file '{file}': {e}")

    def do_logs(self, line):
        if False:
            print('Hello World!')
        'logs <sid> [-l count] [-w]\n        Show the most recent [count] logs for a given scan ID, <sid>.\n        If no count is supplied, all logs are given.\n        If -w is supplied, logs will be streamed to the console until\n        Ctrl-C is entered.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        sid = c[0][0]
        limit = None
        if '-l' in c[0]:
            limit = c[0][c[0].index('-l') + 1]
            if not limit.isdigit():
                self.edprint(f'Invalid result count: {limit}')
                return
            limit = int(limit)
        if '-w' not in c[0]:
            d = self.request(self.ownopts['cli.server_baseurl'] + '/scanlog', post={'id': sid, 'limit': limit})
            if not d:
                return
            j = json.loads(d)
            if len(j) < 1:
                self.dprint('No results.')
                return
            self.send_output(d, line, titles={'0': 'Generated', '1': 'Type', '2': 'Source', '3': 'Message'})
            return
        d = self.request(self.ownopts['cli.server_baseurl'] + '/scanlog', post={'id': sid, 'limit': '1'})
        if not d:
            return
        j = json.loads(d)
        if len(j) < 1:
            self.dprint('No logs (yet?).')
            return
        rowid = j[0][4]
        if not limit:
            limit = 10
        d = self.request(self.ownopts['cli.server_baseurl'] + '/scanlog', post={'id': sid, 'reverse': '1', 'rowId': rowid - limit})
        if not d:
            return
        j = json.loads(d)
        for r in j:
            if r[2] == 'ERROR':
                self.edprint(f'{r[1]}: {r[3]}')
            else:
                self.dprint(f'{r[1]}: {r[3]}')
        try:
            while True:
                d = self.request(self.ownopts['cli.server_baseurl'] + '/scanlog', post={'id': sid, 'reverse': '1', 'rowId': rowid})
                if not d:
                    return
                j = json.loads(d)
                for r in j:
                    if r[2] == 'ERROR':
                        self.edprint(f'{r[1]}: {r[3]}')
                    else:
                        self.dprint(f'{r[1]}: {r[3]}')
                    rowid = str(r[4])
                time.sleep(0.5)
        except KeyboardInterrupt:
            return

    def do_start(self, line):
        if False:
            return 10
        'start <target> (-m m1,... | -t t1,... | -u case) [-n name] [-w]\n        Start a scan against <target> using modules m1,... OR looking\n        for types t1,...\n        OR by use case ("all", "investigate", "passive" and "footprint").\n\n        Scan be be optionally named [name], without a name the target\n        will be used.\n        Use -w to watch the logs from the scan. Ctrl-C to abort the\n        logging (but will not abort the scan).\n        '
        c = self.myparseline(line)
        if len(c[0]) < 3:
            self.edprint('Invalid syntax.')
            return None
        mods = ''
        types = ''
        usecase = ''
        if '-m' in c[0]:
            mods = c[0][c[0].index('-m') + 1]
        if '-t' in c[0]:
            types = c[0][c[0].index('-t') + 1]
        if '-u' in c[0]:
            usecase = c[0][c[0].index('-u') + 1]
        if not mods and (not types) and (not usecase):
            self.edprint('Invalid syntax.')
            return None
        target = c[0][0]
        if '-n' in c[0]:
            title = c[0][c[0].index('-n') + 1]
        else:
            title = target
        post = {'scanname': title, 'scantarget': target, 'modulelist': mods, 'typelist': types, 'usecase': usecase}
        d = self.request(self.ownopts['cli.server_baseurl'] + '/startscan', post=post)
        if not d:
            return None
        s = json.loads(d)
        if s[0] == 'SUCCESS':
            self.dprint('Successfully initiated scan.')
            self.dprint(f'Scan ID: {s[1]}')
        else:
            self.dprint(f'Unable to start scan: {s[1]}')
        if '-w' in c[0]:
            return self.do_logs(f'{s[1]} -w')
        return None

    def do_stop(self, line):
        if False:
            while True:
                i = 10
        'stop <sid>\n        Abort the running scan with scan ID, <sid>.'
        c = self.myparseline(line)
        try:
            scan_id = c[0][0]
        except BaseException:
            self.edprint('Invalid syntax.')
            return
        self.request(self.ownopts['cli.server_baseurl'] + f'/stopscan?id={scan_id}')
        self.dprint(f'Successfully requested scan {id} to stop. This could take some minutes to complete.')

    def do_search(self, line):
        if False:
            while True:
                i = 10
        "search (look up 'find')\n        "
        return self.do_find(line)

    def do_find(self, line):
        if False:
            for i in range(10):
                print('nop')
        'find "<string|/regex/>" <[-s sid]|[-t type]> [-x]\n        Search for string/regex, limited to the scope of either a scan ID or\n        event type. -x for extended format.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        val = c[0][0]
        sid = None
        etype = None
        if '-t' in c[0]:
            etype = c[0][c[0].index('-t') + 1]
        if '-s' in c[0]:
            sid = c[0][c[0].index('-s') + 1]
        titles = {'0': 'Last Seen', '1': 'Data', '3': 'Module'}
        if '-x' in c[0]:
            titles['2'] = 'Source Data'
        d = self.request(self.ownopts['cli.server_baseurl'] + '/search', post={'value': val, 'id': sid, 'eventType': etype})
        if not d:
            return
        j = json.loads(d)
        if not j:
            self.dprint('No results found.')
            return
        if len(j) < 1:
            self.dprint('No results found.')
            return
        self.send_output(d, line, titles)

    def do_summary(self, line):
        if False:
            print('Hello World!')
        'summary <sid> [-t]\n        Summarise the results for a scan ID, <sid>. -t to only show\n        the element types.'
        c = self.myparseline(line)
        if len(c[0]) < 1:
            self.edprint('Invalid syntax.')
            return
        sid = c[0][0]
        if '-t' in c[0]:
            titles = {'0': 'Element Type'}
        else:
            titles = {'0': 'Element Type', '1': 'Element Description', '3': 'Total', '4': 'Unique'}
        d = self.request(self.ownopts['cli.server_baseurl'] + f'/scansummary?id={sid}&by=type')
        if not d:
            return
        j = json.loads(d)
        if not j:
            self.dprint('No results found.')
            return
        if len(j) < 1:
            self.dprint('No results found.')
            return
        self.send_output(d, line, titles, total=False)

    def do_delete(self, line):
        if False:
            return 10
        'delete <sid>\n        Delete a scan with scan ID, <sid>.'
        c = self.myparseline(line)
        try:
            scan_id = c[0][0]
        except BaseException:
            self.edprint('Invalid syntax.')
            return
        self.request(self.ownopts['cli.server_baseurl'] + f'/scandelete?id={scan_id}')
        self.dprint(f'Successfully deleted scan {scan_id}.')

    def print_topics(self, header, cmds, cmdlen, maxcol):
        if False:
            while True:
                i = 10
        if not cmds:
            return
        helpmap = [['help [command]', 'This help output.'], ['debug', 'Enable/Disable debug output.'], ['clear', 'Clear the screen.'], ['history', 'Enable/Disable/List command history.'], ['spool', 'Enable/Disable spooling output.'], ['shell', 'Execute a shell command.'], ['exit', "Exit the SpiderFoot CLI (won't impact running scans)."], ['ping', 'Test connectivity to the SpiderFoot server.'], ['modules', 'List available modules.'], ['types', 'List available data types.'], ['correlationrules', 'List available correlation rules.'], ['set', 'Set variables and configuration settings.'], ['scans', 'List all scans that have been run or are running.'], ['start', 'Start a new scan.'], ['stop', 'Stop a scan.'], ['delete', 'Delete a scan.'], ['scaninfo', 'Scan information.'], ['data', "Show data from a scan's results."], ['export', 'Export scan results to file.'], ['correlations', 'Show correlation results from a scan.'], ['summary', 'Scan result summary.'], ['find', 'Search for data within scan results.'], ['query', 'Run SQL against the SpiderFoot SQLite database.'], ['logs', 'View/watch logs from a scan.']]
        self.send_output(json.dumps(helpmap), '', titles={'0': 'Command', '1': 'Description'}, total=False)

    def do_set(self, line):
        if False:
            for i in range(10):
                print('nop')
        'set [opt [= <val>]]\n        Set a configuration variable in SpiderFoot.'
        c = self.myparseline(line, replace=False)
        cfg = None
        val = None
        if len(c[0]) > 0:
            cfg = c[0][0]
        if len(c[0]) > 2:
            try:
                val = c[0][2]
            except BaseException:
                self.edprint('Invalid syntax.')
                return
        if cfg and val:
            if cfg.startswith('$'):
                self.ownopts[cfg] = val
                self.dprint(f'{cfg} set to {val}')
                return
            if cfg in self.ownopts:
                if isinstance(self.ownopts[cfg], bool):
                    if val.lower() == 'false' or val == '0':
                        val = False
                    else:
                        val = True
                self.ownopts[cfg] = val
                self.dprint(f'{cfg} set to {val}')
                return
        d = self.request(self.ownopts['cli.server_baseurl'] + '/optsraw')
        if not d:
            self.edprint('Unable to obtain SpiderFoot server-side config.')
            return
        j = list()
        serverconfig = dict()
        token = ''
        j = json.loads(d)
        if j[0] == 'ERROR':
            self.edprint('Error fetching SpiderFoot server-side config.')
            return
        serverconfig = j[1]['data']
        token = j[1]['token']
        self.ddprint(str(serverconfig))
        if not cfg or not val:
            ks = list(self.ownopts.keys())
            ks.sort()
            output = list()
            for k in ks:
                c = self.ownopts[k]
                if isinstance(c, bool):
                    c = str(c)
                if not cfg:
                    output.append({'opt': k, 'val': c})
                    continue
                if cfg == k:
                    self.dprint(f'{k} = {c}', plain=True)
            for k in sorted(serverconfig.keys()):
                if type(serverconfig[k]) == list:
                    serverconfig[k] = ','.join(serverconfig[k])
                if not cfg:
                    output.append({'opt': k, 'val': str(serverconfig[k])})
                    continue
                if cfg == k:
                    self.dprint(f'{k} = {serverconfig[k]}', plain=True)
            if len(output) > 0:
                self.send_output(json.dumps(output), line, {'opt': 'Option', 'val': 'Value'}, total=False)
            return
        if val:
            confdata = dict()
            found = False
            for k in serverconfig:
                if k == cfg:
                    serverconfig[k] = val
                    if type(val) == str:
                        if val.lower() == 'true':
                            serverconfig[k] = '1'
                        if val.lower() == 'false':
                            serverconfig[k] = '0'
                    found = True
            if not found:
                self.edprint('Variable not found, so not set.')
                return
            for k in serverconfig:
                optstr = ':'.join(k.split('.')[1:])
                if type(serverconfig[k]) == bool:
                    if serverconfig[k]:
                        confdata[optstr] = '1'
                    else:
                        confdata[optstr] = '0'
                if type(serverconfig[k]) == list:
                    confdata[optstr] = ','.join(serverconfig[k])
                if type(serverconfig[k]) == int:
                    confdata[optstr] = str(serverconfig[k])
                if type(serverconfig[k]) == str:
                    confdata[optstr] = serverconfig[k]
            self.ddprint(str(confdata))
            d = self.request(self.ownopts['cli.server_baseurl'] + '/savesettingsraw', post={'token': token, 'allopts': json.dumps(confdata)})
            j = list()
            if not d:
                self.edprint('Unable to set SpiderFoot server-side config.')
                return
            j = json.loads(d)
            if j[0] == 'ERROR':
                self.edprint(f'Error setting SpiderFoot server-side config: {j[1]}')
                return
            self.dprint(f'{cfg} set to {val}')
            return
        if cfg not in self.ownopts:
            self.edprint('Variable not found, so not set. Did you mean to use a $ variable?')
            return

    def do_shell(self, line):
        if False:
            i = 10
            return i + 15
        'shell\n        Run a shell command locally.'
        self.dprint('Running shell command:' + str(line))
        self.dprint(os.popen(line).read(), plain=True)

    def do_clear(self, line):
        if False:
            return 10
        'clear\n        Clear the screen.'
        sys.stderr.write('\x1b[2J\x1b[H')

    def do_exit(self, line):
        if False:
            print('Hello World!')
        'exit\n        Exit the SpiderFoot CLI.'
        return True

    def do_EOF(self, line):
        if False:
            for i in range(10):
                print('nop')
        'EOF (Ctrl-D)\n        Exit the SpiderFoot CLI.'
        print('\n')
        return True
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='SpiderFoot: Open Source Intelligence Automation.')
    p.add_argument('-d', '--debug', help='Enable debug output.', action='store_true')
    p.add_argument('-s', metavar='URL', type=str, help='Connect to SpiderFoot server on URL. By default, a connection to http://127.0.0.1:5001 will be attempted.')
    p.add_argument('-u', metavar='USER', type=str, help='Username to authenticate to SpiderFoot server.')
    p.add_argument('-p', metavar='PASS', type=str, help="Password to authenticate to SpiderFoot server. Consider using -P PASSFILE instead so that your password isn't visible in your shell history or in process lists!")
    p.add_argument('-P', metavar='PASSFILE', type=str, help='File containing password to authenticate to SpiderFoot server. Ensure permissions on the file are set appropriately!')
    p.add_argument('-e', metavar='FILE', type=str, help='Execute commands from FILE.')
    p.add_argument('-l', metavar='FILE', type=str, help='Log command history to FILE. By default, history is stored to ~/.spiderfoot_history unless disabled with -n.')
    p.add_argument('-n', action='store_true', help='Disable history logging.')
    p.add_argument('-o', metavar='FILE', type=str, help='Spool commands and output to FILE.')
    p.add_argument('-i', help='Allow insecure server connections when using SSL', action='store_true')
    p.add_argument('-q', help='Silent output, only errors reported.', action='store_true')
    p.add_argument('-k', help='Turn off color-coded output.', action='store_true')
    p.add_argument('-b', '-v', help='Print the banner w/ version and exit.', action='store_true')
    args = p.parse_args()
    if args.e:
        try:
            with open(args.e, 'r') as f:
                cin = f.read()
        except BaseException as e:
            print(f'Unable to open {args.e}: ({e})')
            sys.exit(-1)
    else:
        cin = sys.stdin
    s = SpiderFootCli(stdin=cin)
    s.identchars += '$'
    if args.u:
        s.ownopts['cli.username'] = args.u
    if args.p:
        s.ownopts['cli.password'] = args.p
    if args.P:
        try:
            with open(args.P, 'r') as f:
                s.ownopts['cli.password'] = f.readlines()[0].strip('\n')
        except BaseException as e:
            print(f'Unable to open {args.P}: ({e})')
            sys.exit(-1)
    if args.i:
        s.ownopts['cli.ssl_verify'] = False
    if args.k:
        s.ownopts['cli.color'] = False
    if args.s:
        s.ownopts['cli.server_baseurl'] = args.s
    if args.debug:
        s.ownopts['cli.debug'] = True
    if args.q:
        s.ownopts['cli.silent'] = True
    if args.n:
        s.ownopts['cli.history'] = False
    if args.l:
        s.ownopts['cli.history_file'] = args.l
    else:
        try:
            s.ownopts['cli.history_file'] = expanduser('~') + '/.spiderfoot_history'
        except BaseException as e:
            s.dprint(f"Failed to set 'cli.history_file': {e}")
            s.dprint("Using '.spiderfoot_history' in working directory")
            s.ownopts['cli.history_file'] = '.spiderfoot_history'
    if args.o:
        s.ownopts['cli.spool'] = True
        s.ownopts['cli.spool_file'] = args.o
    if args.e or not os.isatty(0):
        try:
            s.use_rawinput = False
            s.prompt = ''
            s.cmdloop()
        finally:
            cin.close()
        sys.exit(0)
    if not args.q:
        s = SpiderFootCli()
        s.dprint(ASCII_LOGO, plain=True, color=bcolors.GREYBLUE)
        s.dprint(COPYRIGHT_INFO, plain=True, color=bcolors.GREYBLUE_DARK)
        s.dprint(f'Version {s.version}.')
        if args.b:
            sys.exit(0)
    s.do_ping('')
    if not args.n:
        try:
            f = codecs.open(s.ownopts['cli.history_file'], 'r', encoding='utf-8')
            for line in f.readlines():
                readline.add_history(line.strip())
            s.dprint('Loaded previous command history.')
        except BaseException:
            pass
    try:
        s.dprint("Type 'help' or '?'.")
        s.cmdloop()
    except KeyboardInterrupt:
        print('\n')
        sys.exit(0)