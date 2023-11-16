"""
This file is part of Commix Project (https://commixproject.com).
Copyright (c) 2014-2023 Anastasios Stasinopoulos (@ancst).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For more see the file 'readme/COPYING' for copying permission.
"""
from src.utils import settings
'\nAbout: Replaces the printf-based ASCII to Decimal `printf "%d" "\'$char\'"` with `echo -n $char | od -An -tuC | xargs`.\nNotes: This tamper script works against Unix-like target(s)\n'
__tamper__ = 'printf2echo'
settings.TAMPER_SCRIPTS[__tamper__] = True

def tamper(payload):
    if False:
        i = 10
        return i + 15

    def printf_to_echo(payload):
        if False:
            while True:
                i = 10
        if 'printf' in payload:
            payload = payload.replace('str=$(printf' + settings.WHITESPACES[0] + "'%d'" + settings.WHITESPACES[0] + '"\'$char\'")', 'str=$(echo' + settings.WHITESPACES[0] + '-n' + settings.WHITESPACES[0] + '$char' + settings.WHITESPACES[0] + '|' + settings.WHITESPACES[0] + 'od' + settings.WHITESPACES[0] + '-An' + settings.WHITESPACES[0] + '-tuC' + settings.WHITESPACES[0] + '|' + settings.WHITESPACES[0] + 'xargs)')
        return payload
    return printf_to_echo(payload)