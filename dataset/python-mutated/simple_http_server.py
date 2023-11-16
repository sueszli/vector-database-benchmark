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
import errno
import socket
from os import curdir, sep
from src.utils import menu
from src.utils import settings
from socket import error as socket_error
from src.thirdparty.colorama import Fore, Back, Style, init
from src.thirdparty.six.moves import _thread as thread
from src.thirdparty.six.moves import http_client as _http_client
from src.thirdparty.six.moves import socketserver as _socketserver
from src.thirdparty.six.moves import BaseHTTPServer as _BaseHTTPServer
'\nValidates IPv4 addresses.\n'

def is_valid_ipv4(ip_addr):
    if False:
        for i in range(10):
            print('nop')
    pattern = re.compile("\n        ^\n        (?:\n          # Dotted variants:\n          (?:\n            # Decimal 1-255 (no leading 0's)\n            [3-9]\\d?|2(?:5[0-5]|[0-4]?\\d)?|1\\d{0,2}\n          |\n            0x0*[0-9a-f]{1,2}  # Hexadecimal 0x0 - 0xFF (possible leading 0's)\n          |\n            0+[1-3]?[0-7]{0,2} # Octal 0 - 0377 (possible leading 0's)\n          )\n          (?:                  # Repeat 0-3 times, separated by a dot\n            \\.\n            (?:\n              [3-9]\\d?|2(?:5[0-5]|[0-4]?\\d)?|1\\d{0,2}\n            |\n              0x0*[0-9a-f]{1,2}\n            |\n              0+[1-3]?[0-7]{0,2}\n            )\n          ){0,3}\n        |\n          0x0*[0-9a-f]{1,8}    # Hexadecimal notation, 0x0 - 0xffffffff\n        |\n          0+[0-3]?[0-7]{0,10}  # Octal notation, 0 - 037777777777\n        |\n          # Decimal notation, 1-4294967295:\n          429496729[0-5]|42949672[0-8]\\d|4294967[01]\\d\\d|429496[0-6]\\d{3}|\n          42949[0-5]\\d{4}|4294[0-8]\\d{5}|429[0-3]\\d{6}|42[0-8]\\d{7}|\n          4[01]\\d{8}|[1-3]\\d{0,9}|[4-9]\\d{0,8}\n        )\n        $\n    ", re.VERBOSE | re.IGNORECASE)
    return pattern.match(ip_addr) is not None

def grab_ip_addr():
    if False:
        i = 10
        return i + 15
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 53))
        s.settimeout(2)
        ip_addr = s.getsockname()[0]
        s.close()
        return ip_addr
    except socket_error as err_msg:
        if errno.ECONNREFUSED:
            warn_msg = 'Internet seems unreachable.'
            print(settings.print_warning_msg(warn_msg))
        else:
            print(settings.print_critical_msg(str(err_msg)) + '\n')
            raise SystemExit()

class Handler(_BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(self):
        if False:
            return 10
        try:
            f = open(self.path)
            self.send_response(_http_client.OK)
            self.send_header(settings.CONNECTION, 'close')
            self.end_headers()
            self.wfile.write(f.read().encode())
            return
        except Exception:
            error_response = settings.APPLICATION + settings.SINGLE_WHITESPACE + settings.VERSION + ' (https://commixproject.com)'
            self.wfile.write(error_response.encode())

    def log_message(self, format, *args):
        if False:
            while True:
                i = 10
        return

class ReusableTCPServer(_socketserver.TCPServer):
    allow_reuse_address = True

def main():
    if False:
        for i in range(10):
            print('nop')
    try:
        connection_refused = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket_error:
        if errno.ECONNREFUSED:
            connection_refused = True
    if connection_refused == False:
        httpd = ReusableTCPServer(('', settings.LOCAL_HTTP_PORT), Handler)
        thread.start_new_thread(httpd.serve_forever, ())