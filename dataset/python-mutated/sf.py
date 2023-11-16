import argparse
import logging
import multiprocessing as mp
import os
import os.path
import random
import signal
import sys
import time
from copy import deepcopy
import cherrypy
import cherrypy_cors
from cherrypy.lib import auth_digest
from sflib import SpiderFoot
from sfscan import startSpiderFootScanner
from sfwebui import SpiderFootWebUi
from spiderfoot import SpiderFootHelpers
from spiderfoot import SpiderFootDb
from spiderfoot import SpiderFootCorrelator
from spiderfoot.logger import logListenerSetup, logWorkerSetup
from spiderfoot import __version__
scanId = None
dbh = None

def main() -> None:
    if False:
        while True:
            i = 10
    sfWebUiConfig = {'host': '127.0.0.1', 'port': 5001, 'root': '/', 'cors_origins': []}
    sfConfig = {'_debug': False, '_maxthreads': 3, '__logging': True, '__outputfilter': None, '_useragent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:62.0) Gecko/20100101 Firefox/62.0', '_dnsserver': '', '_fetchtimeout': 5, '_internettlds': 'https://publicsuffix.org/list/effective_tld_names.dat', '_internettlds_cache': 72, '_genericusers': ','.join(SpiderFootHelpers.usernamesFromWordlists(['generic-usernames'])), '__database': f'{SpiderFootHelpers.dataPath()}/spiderfoot.db', '__modules__': None, '__correlationrules__': None, '_socks1type': '', '_socks2addr': '', '_socks3port': '', '_socks4user': '', '_socks5pwd': ''}
    sfOptdescs = {'_debug': 'Enable debugging?', '_maxthreads': 'Max number of modules to run concurrently', '_useragent': "User-Agent string to use for HTTP requests. Prefix with an '@' to randomly select the User Agent from a file containing user agent strings for each request, e.g. @C:\\useragents.txt or @/home/bob/useragents.txt. Or supply a URL to load the list from there.", '_dnsserver': "Override the default resolver with another DNS server. For example, 8.8.8.8 is Google's open DNS server.", '_fetchtimeout': 'Number of seconds before giving up on a HTTP request.', '_internettlds': 'List of Internet TLDs.', '_internettlds_cache': "Hours to cache the Internet TLD list. This can safely be quite a long time given that the list doesn't change too often.", '_genericusers': 'List of usernames that if found as usernames or as part of e-mail addresses, should be treated differently to non-generics.', '_socks1type': "SOCKS Server Type. Can be '4', '5', 'HTTP' or 'TOR'", '_socks2addr': 'SOCKS Server IP Address.', '_socks3port': 'SOCKS Server TCP Port. Usually 1080 for 4/5, 8080 for HTTP and 9050 for TOR.', '_socks4user': 'SOCKS Username. Valid only for SOCKS4 and SOCKS5 servers.', '_socks5pwd': 'SOCKS Password. Valid only for SOCKS5 servers.', '_modulesenabled': 'Modules enabled for the scan.'}
    args = None
    p = argparse.ArgumentParser(description=f'SpiderFoot {__version__}: Open Source Intelligence Automation.')
    p.add_argument('-d', '--debug', action='store_true', help='Enable debug output.')
    p.add_argument('-l', metavar='IP:port', help='IP and port to listen on.')
    p.add_argument('-m', metavar='mod1,mod2,...', type=str, help='Modules to enable.')
    p.add_argument('-M', '--modules', action='store_true', help='List available modules.')
    p.add_argument('-C', '--correlate', metavar='scanID', help='Run correlation rules against a scan ID.')
    p.add_argument('-s', metavar='TARGET', help='Target for the scan.')
    p.add_argument('-t', metavar='type1,type2,...', type=str, help='Event types to collect (modules selected automatically).')
    p.add_argument('-u', choices=['all', 'footprint', 'investigate', 'passive'], type=str, help='Select modules automatically by use case')
    p.add_argument('-T', '--types', action='store_true', help='List available event types.')
    p.add_argument('-o', choices=['tab', 'csv', 'json'], type=str, help='Output format. Tab is default.')
    p.add_argument('-H', action='store_true', help="Don't print field headers, just data.")
    p.add_argument('-n', action='store_true', help='Strip newlines from data.')
    p.add_argument('-r', action='store_true', help='Include the source data field in tab/csv output.')
    p.add_argument('-S', metavar='LENGTH', type=int, help='Maximum data length to display. By default, all data is shown.')
    p.add_argument('-D', metavar='DELIMITER', type=str, help='Delimiter to use for CSV output. Default is ,.')
    p.add_argument('-f', action='store_true', help="Filter out other event types that weren't requested with -t.")
    p.add_argument('-F', metavar='type1,type2,...', type=str, help='Show only a set of event types, comma-separated.')
    p.add_argument('-x', action='store_true', help='STRICT MODE. Will only enable modules that can directly consume your target, and if -t was specified only those events will be consumed by modules. This overrides -t and -m options.')
    p.add_argument('-q', action='store_true', help='Disable logging. This will also hide errors!')
    p.add_argument('-V', '--version', action='store_true', help='Display the version of SpiderFoot and exit.')
    p.add_argument('-max-threads', type=int, help='Max number of modules to run concurrently.')
    args = p.parse_args()
    if args.version:
        print(f'SpiderFoot {__version__}: Open Source Intelligence Automation.')
        sys.exit(0)
    if args.max_threads:
        sfConfig['_maxthreads'] = args.max_threads
    if args.debug:
        sfConfig['_debug'] = True
    else:
        sfConfig['_debug'] = False
    if args.q:
        sfConfig['__logging'] = False
    loggingQueue = mp.Queue()
    logListenerSetup(loggingQueue, sfConfig)
    logWorkerSetup(loggingQueue)
    log = logging.getLogger(f'spiderfoot.{__name__}')
    sfConfig['__globaloptdescs__'] = sfOptdescs
    try:
        mod_dir = os.path.dirname(os.path.abspath(__file__)) + '/modules/'
        sfModules = SpiderFootHelpers.loadModulesAsDict(mod_dir, ['sfp_template.py'])
    except BaseException as e:
        log.critical(f'Failed to load modules: {e}', exc_info=True)
        sys.exit(-1)
    if not sfModules:
        log.critical(f'No modules found in modules directory: {mod_dir}')
        sys.exit(-1)
    try:
        correlations_dir = os.path.dirname(os.path.abspath(__file__)) + '/correlations/'
        correlationRulesRaw = SpiderFootHelpers.loadCorrelationRulesRaw(correlations_dir, ['template.yaml'])
    except BaseException as e:
        log.critical(f'Failed to load correlation rules: {e}', exc_info=True)
        sys.exit(-1)
    try:
        dbh = SpiderFootDb(sfConfig)
    except Exception as e:
        log.critical(f'Failed to initialize database: {e}', exc_info=True)
        sys.exit(-1)
    sfCorrelationRules = list()
    if not correlationRulesRaw:
        log.error(f'No correlation rules found in correlations directory: {correlations_dir}')
    else:
        try:
            correlator = SpiderFootCorrelator(dbh, correlationRulesRaw)
            sfCorrelationRules = correlator.get_ruleset()
        except Exception as e:
            log.critical(f'Failure initializing correlation rules: {e}', exc_info=True)
            sys.exit(-1)
    sfConfig['__modules__'] = sfModules
    sfConfig['__correlationrules__'] = sfCorrelationRules
    if args.correlate:
        if not correlationRulesRaw:
            log.error('Unable to perform correlations as no correlation rules were found.')
            sys.exit(-1)
        try:
            log.info(f'Running {len(correlationRulesRaw)} correlation rules against scan, {args.correlate}.')
            corr = SpiderFootCorrelator(dbh, correlationRulesRaw, args.correlate)
            corr.run_correlations()
        except Exception as e:
            log.critical(f'Unable to run correlation rules: {e}', exc_info=True)
            sys.exit(-1)
        sys.exit(0)
    if args.modules:
        log.info('Modules available:')
        for m in sorted(sfModules.keys()):
            if '__' in m:
                continue
            print(f"{m.ljust(25)}  {sfModules[m]['descr']}")
        sys.exit(0)
    if args.types:
        dbh = SpiderFootDb(sfConfig, init=True)
        log.info('Types available:')
        typedata = dbh.eventTypes()
        types = dict()
        for r in typedata:
            types[r[1]] = r[0]
        for t in sorted(types.keys()):
            print(f'{t.ljust(45)}  {types[t]}')
        sys.exit(0)
    if args.l:
        try:
            (host, port) = args.l.split(':')
        except BaseException:
            log.critical('Invalid ip:port format.')
            sys.exit(-1)
        sfWebUiConfig['host'] = host
        sfWebUiConfig['port'] = port
        start_web_server(sfWebUiConfig, sfConfig, loggingQueue)
        sys.exit(0)
    start_scan(sfConfig, sfModules, args, loggingQueue)

def start_scan(sfConfig: dict, sfModules: dict, args, loggingQueue) -> None:
    if False:
        print('Hello World!')
    'Start scan\n\n    Args:\n        sfConfig (dict): SpiderFoot config options\n        sfModules (dict): modules\n        args (argparse.Namespace): command line args\n        loggingQueue (Queue): main SpiderFoot logging queue\n    '
    log = logging.getLogger(f'spiderfoot.{__name__}')
    global dbh
    global scanId
    dbh = SpiderFootDb(sfConfig, init=True)
    sf = SpiderFoot(sfConfig)
    if not args.s:
        log.error('You must specify a target when running in scan mode. Try --help for guidance.')
        sys.exit(-1)
    if args.x and (not args.t):
        log.error('-x can only be used with -t. Use --help for guidance.')
        sys.exit(-1)
    if args.x and args.m:
        log.error('-x can only be used with -t and not with -m. Use --help for guidance.')
        sys.exit(-1)
    if args.r and (args.o and args.o not in ['tab', 'csv']):
        log.error('-r can only be used when your output format is tab or csv.')
        sys.exit(-1)
    if args.H and (args.o and args.o not in ['tab', 'csv']):
        log.error('-H can only be used when your output format is tab or csv.')
        sys.exit(-1)
    if args.D and args.o != 'csv':
        log.error('-D can only be used when using the csv output format.')
        sys.exit(-1)
    target = args.s
    if ' ' in target:
        target = f'"{target}"'
    if '.' not in target and (not target.startswith('+')) and ('"' not in target):
        target = f'"{target}"'
    targetType = SpiderFootHelpers.targetTypeFromString(target)
    if not targetType:
        log.error(f'Could not determine target type. Invalid target: {target}')
        sys.exit(-1)
    target = target.strip('"')
    modlist = list()
    if not args.t and (not args.m) and (not args.u):
        log.warning("You didn't specify any modules, types or use case, so all modules will be enabled.")
        for m in list(sfModules.keys()):
            if '__' in m:
                continue
            modlist.append(m)
    signal.signal(signal.SIGINT, handle_abort)
    if args.t:
        types = args.t
        modlist = sf.modulesProducing(types)
        newmods = deepcopy(modlist)
        newmodcpy = deepcopy(newmods)
        while len(newmodcpy) > 0:
            for etype in sf.eventsToModules(newmodcpy):
                xmods = sf.modulesProducing([etype])
                for mod in xmods:
                    if mod not in modlist:
                        modlist.append(mod)
                        newmods.append(mod)
            newmodcpy = deepcopy(newmods)
            newmods = list()
    if args.m:
        modlist = list(filter(None, args.m.split(',')))
    if args.u:
        usecase = args.u[0].upper() + args.u[1:]
        for mod in sfConfig['__modules__']:
            if usecase == 'All' or usecase in sfConfig['__modules__'][mod]['group']:
                modlist.append(mod)
    typedata = dbh.eventTypes()
    types = dict()
    for r in typedata:
        types[r[1]] = r[0]
    sfp__stor_stdout_opts = sfConfig['__modules__']['sfp__stor_stdout']['opts']
    sfp__stor_stdout_opts['_eventtypes'] = types
    if args.f:
        if args.f and (not args.t):
            log.error('You can only use -f with -t. Use --help for guidance.')
            sys.exit(-1)
        sfp__stor_stdout_opts['_showonlyrequested'] = True
    if args.F:
        sfp__stor_stdout_opts['_requested'] = args.F.split(',')
        sfp__stor_stdout_opts['_showonlyrequested'] = True
    if args.o:
        if args.o not in ['tab', 'csv', 'json']:
            log.error("Invalid output format selected. Must be 'tab', 'csv' or 'json'.")
            sys.exit(-1)
        sfp__stor_stdout_opts['_format'] = args.o
    if args.t:
        sfp__stor_stdout_opts['_requested'] = args.t.split(',')
    if args.n:
        sfp__stor_stdout_opts['_stripnewline'] = True
    if args.r:
        sfp__stor_stdout_opts['_showsource'] = True
    if args.S:
        sfp__stor_stdout_opts['_maxlength'] = args.S
    if args.D:
        sfp__stor_stdout_opts['_csvdelim'] = args.D
    if args.x:
        tmodlist = list()
        modlist = list()
        xmods = sf.modulesConsuming([targetType])
        for mod in xmods:
            if mod not in modlist:
                tmodlist.append(mod)
        rtypes = args.t.split(',')
        for mod in tmodlist:
            for r in rtypes:
                if not sfModules[mod]['provides']:
                    continue
                if r in sfModules[mod].get('provides', []) and mod not in modlist:
                    modlist.append(mod)
    if len(modlist) == 0:
        log.error('Based on your criteria, no modules were enabled.')
        sys.exit(-1)
    modlist += ['sfp__stor_db', 'sfp__stor_stdout']
    if sfConfig['__logging']:
        log.info(f"Modules enabled ({len(modlist)}): {','.join(modlist)}")
    cfg = sf.configUnserialize(dbh.configGet(), sfConfig)
    if args.debug:
        cfg['_debug'] = True
    else:
        cfg['_debug'] = False
    if args.x and args.t:
        cfg['__outputfilter'] = args.t.split(',')
    if args.o == 'json':
        print('[', end='')
    elif not args.H:
        delim = '\t'
        if args.o == 'tab':
            delim = '\t'
        if args.o == 'csv':
            if args.D:
                delim = args.D
            else:
                delim = ','
        if args.r:
            if delim == '\t':
                headers = delim.join(['Source'.ljust(30), 'Type'.ljust(45), 'Source Data', 'Data'])
            else:
                headers = delim.join(['Source', 'Type', 'Source Data', 'Data'])
        elif delim == '\t':
            headers = delim.join(['Source'.ljust(30), 'Type'.ljust(45), 'Data'])
        else:
            headers = delim.join(['Source', 'Type', 'Data'])
        print(headers)
    scanName = target
    scanId = SpiderFootHelpers.genScanInstanceId()
    try:
        p = mp.Process(target=startSpiderFootScanner, args=(loggingQueue, scanName, scanId, target, targetType, modlist, cfg))
        p.daemon = True
        p.start()
    except BaseException as e:
        log.error(f'Scan [{scanId}] failed: {e}')
        sys.exit(-1)
    while True:
        time.sleep(1)
        info = dbh.scanInstanceGet(scanId)
        if not info:
            continue
        if info[5] in ['ERROR-FAILED', 'ABORT-REQUESTED', 'ABORTED', 'FINISHED']:
            timeout = 60
            p.join(timeout=timeout)
            if p.is_alive():
                log.error(f'Timeout reached ({timeout}s) waiting for scan {scanId} post-processing to complete.')
                sys.exit(-1)
            if sfConfig['__logging']:
                log.info(f'Scan completed with status {info[5]}')
            if args.o == 'json':
                print(']')
            sys.exit(0)
    return

def start_web_server(sfWebUiConfig: dict, sfConfig: dict, loggingQueue=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Start the web server so you can start looking at results\n\n    Args:\n        sfWebUiConfig (dict): web server options\n        sfConfig (dict): SpiderFoot config options\n        loggingQueue (Queue): main SpiderFoot logging queue\n    '
    log = logging.getLogger(f'spiderfoot.{__name__}')
    web_host = sfWebUiConfig.get('host', '127.0.0.1')
    web_port = sfWebUiConfig.get('port', 5001)
    web_root = sfWebUiConfig.get('root', '/')
    cors_origins = sfWebUiConfig.get('cors_origins', [])
    cherrypy.config.update({'log.screen': False, 'server.socket_host': web_host, 'server.socket_port': int(web_port)})
    log.info(f'Starting web server at {web_host}:{web_port} ...')
    conf = {'/query': {'tools.encode.text_only': False, 'tools.encode.add_charset': True}, '/static': {'tools.staticdir.on': True, 'tools.staticdir.dir': 'static', 'tools.staticdir.root': f'{os.path.dirname(os.path.abspath(__file__))}/spiderfoot'}}
    secrets = dict()
    passwd_file = SpiderFootHelpers.dataPath() + '/passwd'
    if os.path.isfile(passwd_file):
        if not os.access(passwd_file, os.R_OK):
            log.error('Could not read passwd file. Permission denied.')
            sys.exit(-1)
        with open(passwd_file, 'r') as f:
            passwd_data = f.readlines()
        for line in passwd_data:
            if line.strip() == '':
                continue
            if ':' not in line:
                log.error('Incorrect format of passwd file, must be username:password on each line.')
                sys.exit(-1)
            u = line.strip().split(':')[0]
            p = ':'.join(line.strip().split(':')[1:])
            if not u or not p:
                log.error('Incorrect format of passwd file, must be username:password on each line.')
                sys.exit(-1)
            secrets[u] = p
    if secrets:
        log.info('Enabling authentication based on supplied passwd file.')
        conf['/'] = {'tools.auth_digest.on': True, 'tools.auth_digest.realm': web_host, 'tools.auth_digest.get_ha1': auth_digest.get_ha1_dict_plain(secrets), 'tools.auth_digest.key': random.SystemRandom().randint(0, 99999999)}
    else:
        warn_msg = '\n********************************************************************\n'
        warn_msg += 'Warning: passwd file contains no passwords. Authentication disabled.\n'
        warn_msg += 'Please consider adding authentication to protect this instance!\n'
        warn_msg += 'Refer to https://www.spiderfoot.net/documentation/#security.\n'
        warn_msg += '********************************************************************\n'
        log.warning(warn_msg)
    using_ssl = False
    key_path = SpiderFootHelpers.dataPath() + '/spiderfoot.key'
    crt_path = SpiderFootHelpers.dataPath() + '/spiderfoot.crt'
    if os.path.isfile(key_path) and os.path.isfile(crt_path):
        if not os.access(crt_path, os.R_OK):
            log.critical(f'Could not read {crt_path} file. Permission denied.')
            sys.exit(-1)
        if not os.access(key_path, os.R_OK):
            log.critical(f'Could not read {key_path} file. Permission denied.')
            sys.exit(-1)
        log.info('Enabling SSL based on supplied key and certificate file.')
        cherrypy.server.ssl_module = 'builtin'
        cherrypy.server.ssl_certificate = crt_path
        cherrypy.server.ssl_private_key = key_path
        using_ssl = True
    if using_ssl:
        url = 'https://'
    else:
        url = 'http://'
    if web_host == '0.0.0.0':
        url = f'{url}127.0.0.1:{web_port}'
    else:
        url = f'{url}{web_host}:{web_port}{web_root}'
        cors_origins.append(url)
    cherrypy_cors.install()
    cherrypy.config.update({'cors.expose.on': True, 'cors.expose.origins': cors_origins, 'cors.preflight.origins': cors_origins})
    print('')
    print('*************************************************************')
    print(' Use SpiderFoot by starting your web browser of choice and ')
    print(f' browse to {url}')
    print('*************************************************************')
    print('')
    cherrypy.engine.autoreload.unsubscribe()
    cherrypy.quickstart(SpiderFootWebUi(sfWebUiConfig, sfConfig, loggingQueue), script_name=web_root, config=conf)

def handle_abort(signal, frame) -> None:
    if False:
        return 10
    'Handle interrupt and abort scan.\n\n    Args:\n        signal: TBD\n        frame: TBD\n    '
    log = logging.getLogger(f'spiderfoot.{__name__}')
    global dbh
    global scanId
    if scanId and dbh:
        log.info(f'Aborting scan [{scanId}] ...')
        dbh.scanInstanceSet(scanId, None, None, 'ABORTED')
    sys.exit(-1)
if __name__ == '__main__':
    if sys.version_info < (3, 7):
        print('SpiderFoot requires Python 3.7 or higher.')
        sys.exit(-1)
    if len(sys.argv) <= 1:
        print('SpiderFoot requires -l <ip>:<port> to start the web server. Try --help for guidance.')
        sys.exit(-1)
    from pathlib import Path
    if os.path.exists('spiderfoot.db'):
        print(f'ERROR: spiderfoot.db file exists in {os.path.dirname(__file__)}')
        print('SpiderFoot no longer supports loading the spiderfoot.db database from the application directory.')
        print(f'The database is now loaded from your home directory: {Path.home()}/.spiderfoot/spiderfoot.db')
        print(f'This message will go away once you move or remove spiderfoot.db from {os.path.dirname(__file__)}')
        sys.exit(-1)
    from pathlib import Path
    if os.path.exists('passwd'):
        print(f'ERROR: passwd file exists in {os.path.dirname(__file__)}')
        print('SpiderFoot no longer supports loading credentials from the application directory.')
        print(f'The passwd file is now loaded from your home directory: {Path.home()}/.spiderfoot/passwd')
        print(f'This message will go away once you move or remove passwd from {os.path.dirname(__file__)}')
        sys.exit(-1)
    main()