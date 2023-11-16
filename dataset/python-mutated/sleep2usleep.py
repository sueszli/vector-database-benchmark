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
from src.utils import menu
from src.utils import settings
from src.core.injections.controller import checks
'\nAbout: Replaces "sleep" with "usleep" command in the generated payloads.\nNotes: This tamper script works against Unix-like target(s).\nReference: http://man7.org/linux/man-pages/man3/usleep.3.html\n'
__tamper__ = 'sleep2usleep'
if not settings.TAMPER_SCRIPTS[__tamper__]:
    settings.TAMPER_SCRIPTS[__tamper__] = True

def tamper(payload):
    if False:
        while True:
            i = 10

    def sleep_to_usleep(payload):
        if False:
            i = 10
            return i + 15
        settings.TAMPER_SCRIPTS[__tamper__] = True
        for match in re.finditer('sleep' + settings.WHITESPACES[0] + '([1-9]\\d+|[0-9])', payload):
            sleep_to_usleep = 'u' + match.group(0).split(settings.WHITESPACES[0])[0]
            if match.group(0).split(settings.WHITESPACES[0])[1] != '0':
                usleep_delay = match.group(0).split(settings.WHITESPACES[0])[1] + '0' * 6
            else:
                usleep_delay = match.group(0).split(settings.WHITESPACES[0])[1]
            payload = payload.replace(match.group(0), sleep_to_usleep + settings.WHITESPACES[0] + usleep_delay)
        return payload
    if settings.TARGET_OS != settings.OS.WINDOWS:
        if settings.CLASSIC_STATE != False or settings.EVAL_BASED_STATE != False or settings.FILE_BASED_STATE != False:
            if settings.TRANFROM_PAYLOAD == None:
                checks.time_relative_tamper(__tamper__)
                settings.TRANFROM_PAYLOAD = False
            return payload
        else:
            settings.TRANFROM_PAYLOAD = True
            if settings.TRANFROM_PAYLOAD:
                return sleep_to_usleep(payload)
    else:
        return payload