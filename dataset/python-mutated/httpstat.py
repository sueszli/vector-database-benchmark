from __future__ import print_function
import os
import json
import sys
import logging
import tempfile
import subprocess
__version__ = '1.3.2'
PY3 = sys.version_info >= (3,)
if PY3:
    xrange = range

class Env(object):
    prefix = 'HTTPSTAT'
    _instances = []

    def __init__(self, key):
        if False:
            print('Hello World!')
        self.key = key.format(prefix=self.prefix)
        Env._instances.append(self)

    def get(self, default=None):
        if False:
            print('Hello World!')
        return os.environ.get(self.key, default)
ENV_SHOW_BODY = Env('{prefix}_SHOW_BODY')
ENV_SHOW_IP = Env('{prefix}_SHOW_IP')
ENV_SHOW_SPEED = Env('{prefix}_SHOW_SPEED')
ENV_SAVE_BODY = Env('{prefix}_SAVE_BODY')
ENV_CURL_BIN = Env('{prefix}_CURL_BIN')
ENV_METRICS_ONLY = Env('{prefix}_METRICS_ONLY')
ENV_DEBUG = Env('{prefix}_DEBUG')
curl_format = '{\n"time_namelookup": %{time_namelookup},\n"time_connect": %{time_connect},\n"time_appconnect": %{time_appconnect},\n"time_pretransfer": %{time_pretransfer},\n"time_redirect": %{time_redirect},\n"time_starttransfer": %{time_starttransfer},\n"time_total": %{time_total},\n"speed_download": %{speed_download},\n"speed_upload": %{speed_upload},\n"remote_ip": "%{remote_ip}",\n"remote_port": "%{remote_port}",\n"local_ip": "%{local_ip}",\n"local_port": "%{local_port}"\n}'
https_template = '\n  DNS Lookup   TCP Connection   TLS Handshake   Server Processing   Content Transfer\n[   {a0000}  |     {a0001}    |    {a0002}    |      {a0003}      |      {a0004}     ]\n             |                |               |                   |                  |\n    namelookup:{b0000}        |               |                   |                  |\n                        connect:{b0001}       |                   |                  |\n                                    pretransfer:{b0002}           |                  |\n                                                      starttransfer:{b0003}          |\n                                                                                 total:{b0004}\n'[1:]
http_template = '\n  DNS Lookup   TCP Connection   Server Processing   Content Transfer\n[   {a0000}  |     {a0001}    |      {a0003}      |      {a0004}     ]\n             |                |                   |                  |\n    namelookup:{b0000}        |                   |                  |\n                        connect:{b0001}           |                  |\n                                      starttransfer:{b0003}          |\n                                                                 total:{b0004}\n'[1:]
ISATTY = sys.stdout.isatty()

def make_color(code):
    if False:
        for i in range(10):
            print('nop')

    def color_func(s):
        if False:
            i = 10
            return i + 15
        if not ISATTY:
            return s
        tpl = '\x1b[{}m{}\x1b[0m'
        return tpl.format(code, s)
    return color_func
red = make_color(31)
green = make_color(32)
yellow = make_color(33)
blue = make_color(34)
magenta = make_color(35)
cyan = make_color(36)
bold = make_color(1)
underline = make_color(4)
grayscale = {i - 232: make_color('38;5;' + str(i)) for i in xrange(232, 256)}

def quit(s, code=0):
    if False:
        for i in range(10):
            print('nop')
    if s is not None:
        print(s)
    sys.exit(code)

def print_help():
    if False:
        i = 10
        return i + 15
    help = '\nUsage: httpstat URL [CURL_OPTIONS]\n       httpstat -h | --help\n       httpstat --version\n\nArguments:\n  URL     url to request, could be with or without `http(s)://` prefix\n\nOptions:\n  CURL_OPTIONS  any curl supported options, except for -w -D -o -S -s,\n                which are already used internally.\n  -h --help     show this screen.\n  --version     show version.\n\nEnvironments:\n  HTTPSTAT_SHOW_BODY    Set to `true` to show response body in the output,\n                        note that body length is limited to 1023 bytes, will be\n                        truncated if exceeds. Default is `false`.\n  HTTPSTAT_SHOW_IP      By default httpstat shows remote and local IP/port address.\n                        Set to `false` to disable this feature. Default is `true`.\n  HTTPSTAT_SHOW_SPEED   Set to `true` to show download and upload speed.\n                        Default is `false`.\n  HTTPSTAT_SAVE_BODY    By default httpstat stores body in a tmp file,\n                        set to `false` to disable this feature. Default is `true`\n  HTTPSTAT_CURL_BIN     Indicate the curl bin path to use. Default is `curl`\n                        from current shell $PATH.\n  HTTPSTAT_DEBUG        Set to `true` to see debugging logs. Default is `false`\n'[1:-1]
    print(help)

def main():
    if False:
        for i in range(10):
            print('nop')
    args = sys.argv[1:]
    if not args:
        print_help()
        quit(None, 0)
    show_body = 'true' in ENV_SHOW_BODY.get('false').lower()
    show_ip = 'true' in ENV_SHOW_IP.get('true').lower()
    show_speed = 'true' in ENV_SHOW_SPEED.get('false').lower()
    save_body = 'true' in ENV_SAVE_BODY.get('true').lower()
    curl_bin = ENV_CURL_BIN.get('curl')
    metrics_only = 'true' in ENV_METRICS_ONLY.get('false').lower()
    is_debug = 'true' in ENV_DEBUG.get('false').lower()
    if is_debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    lg = logging.getLogger('httpstat')
    lg.debug('Envs:\n%s', '\n'.join(('  {}={}'.format(i.key, i.get('')) for i in Env._instances)))
    lg.debug('Flags: %s', dict(show_body=show_body, show_ip=show_ip, show_speed=show_speed, save_body=save_body, curl_bin=curl_bin, is_debug=is_debug))
    url = args[0]
    if url in ['-h', '--help']:
        print_help()
        quit(None, 0)
    elif url == '--version':
        print('httpstat {}'.format(__version__))
        quit(None, 0)
    curl_args = args[1:]
    exclude_options = ['-w', '--write-out', '-D', '--dump-header', '-o', '--output', '-s', '--silent']
    for i in exclude_options:
        if i in curl_args:
            quit(yellow('Error: {} is not allowed in extra curl args'.format(i)), 1)
    bodyf = tempfile.NamedTemporaryFile(delete=False)
    bodyf.close()
    headerf = tempfile.NamedTemporaryFile(delete=False)
    headerf.close()
    cmd_env = os.environ.copy()
    cmd_env.update(LC_ALL='C')
    cmd_core = [curl_bin, '-w', curl_format, '-D', headerf.name, '-o', bodyf.name, '-s', '-S']
    cmd = cmd_core + curl_args + [url]
    lg.debug('cmd: %s', cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=cmd_env)
    (out, err) = p.communicate()
    if PY3:
        (out, err) = (out.decode(), err.decode())
    lg.debug('out: %s', out)
    if p.returncode == 0:
        if err:
            print(grayscale[16](err))
    else:
        _cmd = list(cmd)
        _cmd[2] = '<output-format>'
        _cmd[4] = '<tempfile>'
        _cmd[6] = '<tempfile>'
        print('> {}'.format(' '.join(_cmd)))
        quit(yellow('curl error: {}'.format(err)), p.returncode)
    try:
        d = json.loads(out)
    except ValueError as e:
        print(yellow('Could not decode json: {}'.format(e)))
        print('curl result:', p.returncode, grayscale[16](out), grayscale[16](err))
        quit(None, 1)
    for k in d:
        if k.startswith('time_'):
            v = d[k]
            if isinstance(v, float):
                d[k] = int(v * 1000)
            elif isinstance(v, int):
                d[k] = int(v / 1000)
            else:
                raise TypeError('{} value type is invalid: {}'.format(k, type(v)))
    d.update(range_dns=d['time_namelookup'], range_connection=d['time_connect'] - d['time_namelookup'], range_ssl=d['time_pretransfer'] - d['time_connect'], range_server=d['time_starttransfer'] - d['time_pretransfer'], range_transfer=d['time_total'] - d['time_starttransfer'])
    if metrics_only:
        print(json.dumps(d, indent=2))
        quit(None, 0)
    if show_ip:
        s = 'Connected to {}:{} from {}:{}'.format(cyan(d['remote_ip']), cyan(d['remote_port']), d['local_ip'], d['local_port'])
        print(s)
        print()
    with open(headerf.name, 'r') as f:
        headers = f.read().strip()
    lg.debug('rm header file %s', headerf.name)
    os.remove(headerf.name)
    for (loop, line) in enumerate(headers.split('\n')):
        if loop == 0:
            (p1, p2) = tuple(line.split('/'))
            print(green(p1) + grayscale[14]('/') + cyan(p2))
        else:
            pos = line.find(':')
            print(grayscale[14](line[:pos + 1]) + cyan(line[pos + 1:]))
    print()
    if show_body:
        body_limit = 1024
        with open(bodyf.name, 'r') as f:
            body = f.read().strip()
        body_len = len(body)
        if body_len > body_limit:
            print(body[:body_limit] + cyan('...'))
            print()
            s = '{} is truncated ({} out of {})'.format(green('Body'), body_limit, body_len)
            if save_body:
                s += ', stored in: {}'.format(bodyf.name)
            print(s)
        else:
            print(body)
    elif save_body:
        print('{} stored in: {}'.format(green('Body'), bodyf.name))
    if not save_body:
        lg.debug('rm body file %s', bodyf.name)
        os.remove(bodyf.name)
    if url.startswith('https://'):
        template = https_template
    else:
        template = http_template
    tpl_parts = template.split('\n')
    tpl_parts[0] = grayscale[16](tpl_parts[0])
    template = '\n'.join(tpl_parts)

    def fmta(s):
        if False:
            return 10
        return cyan('{:^7}'.format(str(s) + 'ms'))

    def fmtb(s):
        if False:
            print('Hello World!')
        return cyan('{:<7}'.format(str(s) + 'ms'))
    stat = template.format(a0000=fmta(d['range_dns']), a0001=fmta(d['range_connection']), a0002=fmta(d['range_ssl']), a0003=fmta(d['range_server']), a0004=fmta(d['range_transfer']), b0000=fmtb(d['time_namelookup']), b0001=fmtb(d['time_connect']), b0002=fmtb(d['time_pretransfer']), b0003=fmtb(d['time_starttransfer']), b0004=fmtb(d['time_total']))
    print()
    print(stat)
    if show_speed:
        print('speed_download: {:.1f} KiB/s, speed_upload: {:.1f} KiB/s'.format(d['speed_download'] / 1024, d['speed_upload'] / 1024))
if __name__ == '__main__':
    main()