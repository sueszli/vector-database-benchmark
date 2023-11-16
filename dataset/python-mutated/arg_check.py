import re
import sys
DEPRECATED = '\n--confdir\n-Z\n--body-size-limit\n--stream\n--palette\n--palette-transparent\n--follow\n--order\n--no-mouse\n--reverse\n--http2-priority\n--no-http2-priority\n--no-websocket\n--websocket\n--upstream-bind-address\n--ciphers-client\n--ciphers-server\n--client-certs\n--no-upstream-cert\n--add-upstream-certs-to-client-chain\n--upstream-trusted-confdir\n--upstream-trusted-ca\n--ssl-version-client\n--ssl-version-server\n--no-onboarding\n--onboarding-host\n--onboarding-port\n--server-replay-use-header\n--no-pop\n--replay-ignore-content\n--replay-ignore-payload-param\n--replay-ignore-param\n--replay-ignore-host\n--replace-from-file\n'
REPLACED = '\n-t\n-u\n--wfile\n-a\n--afile\n-z\n-b\n--bind-address\n--port\n-I\n--ignore\n--tcp\n--cert\n--insecure\n-c\n--replace\n--replacements\n-i\n-f\n--filter\n--socks\n--server-replay-nopop\n'
REPLACEMENTS = {'--stream': 'stream_large_bodies', '--palette': 'console_palette', '--palette-transparent': 'console_palette_transparent:', '--follow': 'console_focus_follow', '--order': 'view_order', '--no-mouse': 'console_mouse', '--reverse': 'view_order_reversed', '--no-websocket': 'websocket', '--no-upstream-cert': 'upstream_cert', '--upstream-trusted-confdir': 'ssl_verify_upstream_trusted_confdir', '--upstream-trusted-ca': 'ssl_verify_upstream_trusted_ca', '--no-onboarding': 'onboarding', '--no-pop': 'server_replay_reuse', '--replay-ignore-content': 'server_replay_ignore_content', '--replay-ignore-payload-param': 'server_replay_ignore_payload_params', '--replay-ignore-param': 'server_replay_ignore_params', '--replay-ignore-host': 'server_replay_ignore_host', '--replace-from-file': 'replacements (use @ to specify path)', '-t': '--stickycookie', '-u': '--stickyauth', '--wfile': '--save-stream-file', '-a': '-w  Prefix path with + to append.', '--afile': '-w  Prefix path with + to append.', '-z': '--anticomp', '-b': '--listen-host', '--bind-address': '--listen-host', '--port': '--listen-port', '-I': '--ignore-hosts', '--ignore': '--ignore-hosts', '--tcp': '--tcp-hosts', '--cert': '--certs', '--insecure': '--ssl-insecure', '-c': '-C', '--replace': ['--modify-body', '--modify-headers'], '--replacements': ['--modify-body', '--modify-headers'], '-i': '--intercept', '-f': '--view-filter', '--filter': '--view-filter', '--socks': '--mode socks5', '--server-replay-nopop': '--server-replay-reuse'}

def check():
    if False:
        return 10
    args = sys.argv[1:]
    print()
    if '-U' in args:
        print('-U is deprecated, please use --mode upstream:SPEC instead')
    if '-T' in args:
        print('-T is deprecated, please use --mode transparent instead')
    for option in ('-e', '--eventlog', '--norefresh'):
        if option in args:
            print(f'{option} has been removed.')
    for option in ('--nonanonymous', '--singleuser', '--htpasswd'):
        if option in args:
            print('{} is deprecated.\nPlease use `--proxyauth SPEC` instead.\nSPEC Format: "username:pass", "any" to accept any user/pass combination,\n"@path" to use an Apache htpasswd file, or\n"ldap[s]:url_server_ldap[:port]:dn_auth:password:dn_subtree[?search_filter_key=...]" for LDAP authentication.'.format(option))
    for option in REPLACED.splitlines():
        if option in args:
            r = REPLACEMENTS.get(option)
            if isinstance(r, list):
                new_options = r
            else:
                new_options = [r]
            print('{} is deprecated.\nPlease use `{}` instead.'.format(option, '` or `'.join(new_options)))
    for option in DEPRECATED.splitlines():
        if option in args:
            print('{} is deprecated.\nPlease use `--set {}=value` instead.\nTo show all options and their default values use --options'.format(option, REPLACEMENTS.get(option, None) or option.lstrip('-').replace('-', '_')))
    for argument in args:
        underscoreParam = re.search('[-]{2}((.*?_)(.*?(\\s|$)))+', argument)
        if underscoreParam is not None:
            print('{} uses underscores, please use hyphens {}'.format(argument, argument.replace('_', '-')))