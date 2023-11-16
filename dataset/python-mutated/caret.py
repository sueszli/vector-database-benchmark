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
'\nAbout: Adds caret symbol (^) between the characters of the generated payloads.\nNotes: This tamper script works against windows targets.\n'
__tamper__ = 'caret'
if not settings.TAMPER_SCRIPTS[__tamper__]:
    settings.TAMPER_SCRIPTS[__tamper__] = True

def tamper(payload):
    if False:
        return 10

    def add_caret_symbol(payload):
        if False:
            return 10
        settings.TAMPER_SCRIPTS[__tamper__] = True
        if re.compile('\\w+').findall(payload):
            long_string = ''
            if len(max(re.compile('\\w+').findall(payload), key=lambda word: len(word))) >= 5000:
                long_string = max(re.compile('\\w+').findall(payload), key=lambda word: len(word))
        rep = {'^^': '^', '"^t""^o""^k""^e""^n""^s"': '"t"^"o"^"k"^"e"^"n"^"s"', '^t^o^k^e^n^s': '"t"^"o"^"k"^"e"^"n"^"s"', re.sub('([b-zD-Z])', '^\\1', long_string): long_string.replace('^', '')}
        payload = re.sub('([b-zD-Z])', '^\\1', payload)
        rep = dict(((re.escape(k), v) for (k, v) in rep.items()))
        pattern = re.compile('|'.join(rep.keys()))
        payload = pattern.sub(lambda m: rep[re.escape(m.group(0))], payload)
        return payload
    if settings.TARGET_OS == settings.OS.WINDOWS:
        if settings.EVAL_BASED_STATE != False:
            return payload
        else:
            return add_caret_symbol(payload)
    else:
        return payload