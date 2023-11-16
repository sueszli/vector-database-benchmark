"""
Sherlock: Find Usernames Across Social Networks Module

This module contains the main logic to search for usernames at social
networks.
"""
import csv
import signal
import pandas as pd
import os
import platform
import re
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from time import monotonic
import requests
from requests_futures.sessions import FuturesSession
from torrequest import TorRequest
from result import QueryStatus
from result import QueryResult
from notify import QueryNotifyPrint
from sites import SitesInformation
from colorama import init
module_name = 'Sherlock: Find Usernames Across Social Networks'
__version__ = '0.14.3'

class SherlockFuturesSession(FuturesSession):

    def request(self, method, url, hooks=None, *args, **kwargs):
        if False:
            print('Hello World!')
        'Request URL.\n\n        This extends the FuturesSession request method to calculate a response\n        time metric to each request.\n\n        It is taken (almost) directly from the following Stack Overflow answer:\n        https://github.com/ross/requests-futures#working-in-the-background\n\n        Keyword Arguments:\n        self                   -- This object.\n        method                 -- String containing method desired for request.\n        url                    -- String containing URL for request.\n        hooks                  -- Dictionary containing hooks to execute after\n                                  request finishes.\n        args                   -- Arguments.\n        kwargs                 -- Keyword arguments.\n\n        Return Value:\n        Request object.\n        '
        if hooks is None:
            hooks = {}
        start = monotonic()

        def response_time(resp, *args, **kwargs):
            if False:
                return 10
            'Response Time Hook.\n\n            Keyword Arguments:\n            resp                   -- Response object.\n            args                   -- Arguments.\n            kwargs                 -- Keyword arguments.\n\n            Return Value:\n            Nothing.\n            '
            resp.elapsed = monotonic() - start
            return
        try:
            if isinstance(hooks['response'], list):
                hooks['response'].insert(0, response_time)
            elif isinstance(hooks['response'], tuple):
                hooks['response'] = list(hooks['response'])
                hooks['response'].insert(0, response_time)
            else:
                hooks['response'] = [response_time, hooks['response']]
        except KeyError:
            hooks['response'] = [response_time]
        return super(SherlockFuturesSession, self).request(method, url, *args, hooks=hooks, **kwargs)

def get_response(request_future, error_type, social_network):
    if False:
        return 10
    response = None
    error_context = 'General Unknown Error'
    exception_text = None
    try:
        response = request_future.result()
        if response.status_code:
            error_context = None
    except requests.exceptions.HTTPError as errh:
        error_context = 'HTTP Error'
        exception_text = str(errh)
    except requests.exceptions.ProxyError as errp:
        error_context = 'Proxy Error'
        exception_text = str(errp)
    except requests.exceptions.ConnectionError as errc:
        error_context = 'Error Connecting'
        exception_text = str(errc)
    except requests.exceptions.Timeout as errt:
        error_context = 'Timeout Error'
        exception_text = str(errt)
    except requests.exceptions.RequestException as err:
        error_context = 'Unknown Error'
        exception_text = str(err)
    return (response, error_context, exception_text)

def interpolate_string(object, username):
    if False:
        return 10
    'Insert a string into the string properties of an object recursively.'
    if isinstance(object, str):
        return object.replace('{}', username)
    elif isinstance(object, dict):
        for (key, value) in object.items():
            object[key] = interpolate_string(value, username)
    elif isinstance(object, list):
        for i in object:
            object[i] = interpolate_string(object[i], username)
    return object

def CheckForParameter(username):
    if False:
        while True:
            i = 10
    'checks if {?} exists in the username\n    if exist it means that sherlock is looking for more multiple username'
    return '{?}' in username
checksymbols = []
checksymbols = ['_', '-', '.']

def MultipleUsernames(username):
    if False:
        return 10
    'replace the parameter with with symbols and return a list of usernames'
    allUsernames = []
    for i in checksymbols:
        allUsernames.append(username.replace('{?}', i))
    return allUsernames

def sherlock(username, site_data, query_notify, tor=False, unique_tor=False, proxy=None, timeout=60):
    if False:
        while True:
            i = 10
    'Run Sherlock Analysis.\n\n    Checks for existence of username on various social media sites.\n\n    Keyword Arguments:\n    username               -- String indicating username that report\n                              should be created against.\n    site_data              -- Dictionary containing all of the site data.\n    query_notify           -- Object with base type of QueryNotify().\n                              This will be used to notify the caller about\n                              query results.\n    tor                    -- Boolean indicating whether to use a tor circuit for the requests.\n    unique_tor             -- Boolean indicating whether to use a new tor circuit for each request.\n    proxy                  -- String indicating the proxy URL\n    timeout                -- Time in seconds to wait before timing out request.\n                              Default is 60 seconds.\n\n    Return Value:\n    Dictionary containing results from report. Key of dictionary is the name\n    of the social network site, and the value is another dictionary with\n    the following keys:\n        url_main:      URL of main site.\n        url_user:      URL of user on site (if account exists).\n        status:        QueryResult() object indicating results of test for\n                       account existence.\n        http_status:   HTTP status code of query which checked for existence on\n                       site.\n        response_text: Text that came back from request.  May be None if\n                       there was an HTTP error when checking for existence.\n    '
    query_notify.start(username)
    if tor or unique_tor:
        underlying_request = TorRequest()
        underlying_session = underlying_request.session
    else:
        underlying_session = requests.session()
        underlying_request = requests.Request()
    if len(site_data) >= 20:
        max_workers = 20
    else:
        max_workers = len(site_data)
    session = SherlockFuturesSession(max_workers=max_workers, session=underlying_session)
    results_total = {}
    for (social_network, net_info) in site_data.items():
        results_site = {'url_main': net_info.get('urlMain')}
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'}
        if 'headers' in net_info:
            headers.update(net_info['headers'])
        url = interpolate_string(net_info['url'], username)
        regex_check = net_info.get('regexCheck')
        if regex_check and re.search(regex_check, username) is None:
            results_site['status'] = QueryResult(username, social_network, url, QueryStatus.ILLEGAL)
            results_site['url_user'] = ''
            results_site['http_status'] = ''
            results_site['response_text'] = ''
            query_notify.update(results_site['status'])
        else:
            results_site['url_user'] = url
            url_probe = net_info.get('urlProbe')
            request_method = net_info.get('request_method')
            request_payload = net_info.get('request_payload')
            request = None
            if request_method is not None:
                if request_method == 'GET':
                    request = session.get
                elif request_method == 'HEAD':
                    request = session.head
                elif request_method == 'POST':
                    request = session.post
                elif request_method == 'PUT':
                    request = session.put
                else:
                    raise RuntimeError(f'Unsupported request_method for {url}')
            if request_payload is not None:
                request_payload = interpolate_string(request_payload, username)
            if url_probe is None:
                url_probe = url
            else:
                url_probe = interpolate_string(url_probe, username)
            if request is None:
                if net_info['errorType'] == 'status_code':
                    request = session.head
                else:
                    request = session.get
            if net_info['errorType'] == 'response_url':
                allow_redirects = False
            else:
                allow_redirects = True
            if proxy is not None:
                proxies = {'http': proxy, 'https': proxy}
                future = request(url=url_probe, headers=headers, proxies=proxies, allow_redirects=allow_redirects, timeout=timeout, json=request_payload)
            else:
                future = request(url=url_probe, headers=headers, allow_redirects=allow_redirects, timeout=timeout, json=request_payload)
            net_info['request_future'] = future
            if unique_tor:
                underlying_request.reset_identity()
        results_total[social_network] = results_site
    for (social_network, net_info) in site_data.items():
        results_site = results_total.get(social_network)
        url = results_site.get('url_user')
        status = results_site.get('status')
        if status is not None:
            continue
        error_type = net_info['errorType']
        error_code = net_info.get('errorCode')
        future = net_info['request_future']
        (r, error_text, exception_text) = get_response(request_future=future, error_type=error_type, social_network=social_network)
        try:
            response_time = r.elapsed
        except AttributeError:
            response_time = None
        try:
            http_status = r.status_code
        except:
            http_status = '?'
        try:
            response_text = r.text.encode(r.encoding or 'UTF-8')
        except:
            response_text = ''
        query_status = QueryStatus.UNKNOWN
        error_context = None
        if error_text is not None:
            error_context = error_text
        elif error_type == 'message':
            error_flag = True
            errors = net_info.get('errorMsg')
            if isinstance(errors, str):
                if errors in r.text:
                    error_flag = False
            else:
                for error in errors:
                    if error in r.text:
                        error_flag = False
                        break
            if error_flag:
                query_status = QueryStatus.CLAIMED
            else:
                query_status = QueryStatus.AVAILABLE
        elif error_type == 'status_code':
            if error_code == r.status_code:
                query_status = QueryStatus.AVAILABLE
            elif not r.status_code >= 300 or r.status_code < 200:
                query_status = QueryStatus.CLAIMED
            else:
                query_status = QueryStatus.AVAILABLE
        elif error_type == 'response_url':
            if 200 <= r.status_code < 300:
                query_status = QueryStatus.CLAIMED
            else:
                query_status = QueryStatus.AVAILABLE
        else:
            raise ValueError(f"Unknown Error Type '{error_type}' for site '{social_network}'")
        result = QueryResult(username=username, site_name=social_network, site_url_user=url, status=query_status, query_time=response_time, context=error_context)
        query_notify.update(result)
        results_site['status'] = result
        results_site['http_status'] = http_status
        results_site['response_text'] = response_text
        results_total[social_network] = results_site
    return results_total

def timeout_check(value):
    if False:
        for i in range(10):
            print('nop')
    'Check Timeout Argument.\n\n    Checks timeout for validity.\n\n    Keyword Arguments:\n    value                  -- Time in seconds to wait before timing out request.\n\n    Return Value:\n    Floating point number representing the time (in seconds) that should be\n    used for the timeout.\n\n    NOTE:  Will raise an exception if the timeout in invalid.\n    '
    from argparse import ArgumentTypeError
    try:
        timeout = float(value)
    except:
        raise ArgumentTypeError(f"Timeout '{value}' must be a number.")
    if timeout <= 0:
        raise ArgumentTypeError(f"Timeout '{value}' must be greater than 0.0s.")
    return timeout

def handler(signal_received, frame):
    if False:
        print('Hello World!')
    'Exit gracefully without throwing errors\n\n    Source: https://www.devdungeon.com/content/python-catch-sigint-ctrl-c\n    '
    sys.exit(0)

def main():
    if False:
        print('Hello World!')
    version_string = f'%(prog)s {__version__}\n' + f'{requests.__description__}:  {requests.__version__}\n' + f'Python:  {platform.python_version()}'
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=f'{module_name} (Version {__version__})')
    parser.add_argument('--version', action='version', version=version_string, help='Display version information and dependencies.')
    parser.add_argument('--verbose', '-v', '-d', '--debug', action='store_true', dest='verbose', default=False, help='Display extra debugging information and metrics.')
    parser.add_argument('--folderoutput', '-fo', dest='folderoutput', help='If using multiple usernames, the output of the results will be saved to this folder.')
    parser.add_argument('--output', '-o', dest='output', help='If using single username, the output of the result will be saved to this file.')
    parser.add_argument('--tor', '-t', action='store_true', dest='tor', default=False, help='Make requests over Tor; increases runtime; requires Tor to be installed and in system path.')
    parser.add_argument('--unique-tor', '-u', action='store_true', dest='unique_tor', default=False, help='Make requests over Tor with new Tor circuit after each request; increases runtime; requires Tor to be installed and in system path.')
    parser.add_argument('--csv', action='store_true', dest='csv', default=False, help='Create Comma-Separated Values (CSV) File.')
    parser.add_argument('--xlsx', action='store_true', dest='xlsx', default=False, help='Create the standard file for the modern Microsoft Excel spreadsheet (xslx).')
    parser.add_argument('--site', action='append', metavar='SITE_NAME', dest='site_list', default=None, help='Limit analysis to just the listed sites. Add multiple options to specify more than one site.')
    parser.add_argument('--proxy', '-p', metavar='PROXY_URL', action='store', dest='proxy', default=None, help='Make requests over a proxy. e.g. socks5://127.0.0.1:1080')
    parser.add_argument('--json', '-j', metavar='JSON_FILE', dest='json_file', default=None, help='Load data from a JSON file or an online, valid, JSON file.')
    parser.add_argument('--timeout', action='store', metavar='TIMEOUT', dest='timeout', type=timeout_check, default=60, help='Time (in seconds) to wait for response to requests (Default: 60)')
    parser.add_argument('--print-all', action='store_true', dest='print_all', default=False, help='Output sites where the username was not found.')
    parser.add_argument('--print-found', action='store_true', dest='print_found', default=True, help='Output sites where the username was found (also if exported as file).')
    parser.add_argument('--no-color', action='store_true', dest='no_color', default=False, help="Don't color terminal output")
    parser.add_argument('username', nargs='+', metavar='USERNAMES', action='store', help="One or more usernames to check with social networks. Check similar usernames using {%%} (replace to '_', '-', '.').")
    parser.add_argument('--browse', '-b', action='store_true', dest='browse', default=False, help='Browse to all results on default browser.')
    parser.add_argument('--local', '-l', action='store_true', default=False, help='Force the use of the local data.json file.')
    parser.add_argument('--nsfw', action='store_true', default=False, help='Include checking of NSFW sites from default list.')
    args = parser.parse_args()
    signal.signal(signal.SIGINT, handler)
    try:
        r = requests.get('https://raw.githubusercontent.com/sherlock-project/sherlock/master/sherlock/sherlock.py')
        remote_version = str(re.findall('__version__ = "(.*)"', r.text)[0])
        local_version = __version__
        if remote_version != local_version:
            print('Update Available!\n' + f'You are running version {local_version}. Version {remote_version} is available at https://github.com/sherlock-project/sherlock')
    except Exception as error:
        print(f'A problem occurred while checking for an update: {error}')
    if args.tor and args.proxy is not None:
        raise Exception('Tor and Proxy cannot be set at the same time.')
    if args.proxy is not None:
        print('Using the proxy: ' + args.proxy)
    if args.tor or args.unique_tor:
        print('Using Tor to make requests')
        print('Warning: some websites might refuse connecting over Tor, so note that using this option might increase connection errors.')
    if args.no_color:
        init(strip=True, convert=False)
    else:
        init(autoreset=True)
    if args.output is not None and args.folderoutput is not None:
        print('You can only use one of the output methods.')
        sys.exit(1)
    if args.output is not None and len(args.username) != 1:
        print('You can only use --output with a single username')
        sys.exit(1)
    try:
        if args.local:
            sites = SitesInformation(os.path.join(os.path.dirname(__file__), 'resources/data.json'))
        else:
            sites = SitesInformation(args.json_file)
    except Exception as error:
        print(f'ERROR:  {error}')
        sys.exit(1)
    if not args.nsfw:
        sites.remove_nsfw_sites()
    site_data_all = {site.name: site.information for site in sites}
    if args.site_list is None:
        site_data = site_data_all
    else:
        site_data = {}
        site_missing = []
        for site in args.site_list:
            counter = 0
            for existing_site in site_data_all:
                if site.lower() == existing_site.lower():
                    site_data[existing_site] = site_data_all[existing_site]
                    counter += 1
            if counter == 0:
                site_missing.append(f"'{site}'")
        if site_missing:
            print(f"Error: Desired sites not found: {', '.join(site_missing)}.")
        if not site_data:
            sys.exit(1)
    query_notify = QueryNotifyPrint(result=None, verbose=args.verbose, print_all=args.print_all, browse=args.browse)
    all_usernames = []
    for username in args.username:
        if CheckForParameter(username):
            for name in MultipleUsernames(username):
                all_usernames.append(name)
        else:
            all_usernames.append(username)
    for username in all_usernames:
        results = sherlock(username, site_data, query_notify, tor=args.tor, unique_tor=args.unique_tor, proxy=args.proxy, timeout=args.timeout)
        if args.output:
            result_file = args.output
        elif args.folderoutput:
            os.makedirs(args.folderoutput, exist_ok=True)
            result_file = os.path.join(args.folderoutput, f'{username}.txt')
        else:
            result_file = f'{username}.txt'
        with open(result_file, 'w', encoding='utf-8') as file:
            exists_counter = 0
            for website_name in results:
                dictionary = results[website_name]
                if dictionary.get('status').status == QueryStatus.CLAIMED:
                    exists_counter += 1
                    file.write(dictionary['url_user'] + '\n')
            file.write(f'Total Websites Username Detected On : {exists_counter}\n')
        if args.csv:
            result_file = f'{username}.csv'
            if args.folderoutput:
                os.makedirs(args.folderoutput, exist_ok=True)
                result_file = os.path.join(args.folderoutput, result_file)
            with open(result_file, 'w', newline='', encoding='utf-8') as csv_report:
                writer = csv.writer(csv_report)
                writer.writerow(['username', 'name', 'url_main', 'url_user', 'exists', 'http_status', 'response_time_s'])
                for site in results:
                    if args.print_found and (not args.print_all) and (results[site]['status'].status != QueryStatus.CLAIMED):
                        continue
                    response_time_s = results[site]['status'].query_time
                    if response_time_s is None:
                        response_time_s = ''
                    writer.writerow([username, site, results[site]['url_main'], results[site]['url_user'], str(results[site]['status'].status), results[site]['http_status'], response_time_s])
        if args.xlsx:
            usernames = []
            names = []
            url_main = []
            url_user = []
            exists = []
            http_status = []
            response_time_s = []
            for site in results:
                if args.print_found and (not args.print_all) and (results[site]['status'].status != QueryStatus.CLAIMED):
                    continue
                if response_time_s is None:
                    response_time_s.append('')
                else:
                    response_time_s.append(results[site]['status'].query_time)
                usernames.append(username)
                names.append(site)
                url_main.append(results[site]['url_main'])
                url_user.append(results[site]['url_user'])
                exists.append(str(results[site]['status'].status))
                http_status.append(results[site]['http_status'])
            DataFrame = pd.DataFrame({'username': usernames, 'name': names, 'url_main': url_main, 'url_user': url_user, 'exists': exists, 'http_status': http_status, 'response_time_s': response_time_s})
            DataFrame.to_excel(f'{username}.xlsx', sheet_name='sheet1', index=False)
        print()
    query_notify.finish()
if __name__ == '__main__':
    main()