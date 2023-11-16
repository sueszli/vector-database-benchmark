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
from src.utils import settings
'\nAbout: Adds dollar sign followed by an at-sign ($@) between the characters of the generated payloads.\nNotes: This tamper script works against Unix-like target(s).\n'
__tamper__ = 'dollaratsigns'
if not settings.TAMPER_SCRIPTS[__tamper__]:
    settings.TAMPER_SCRIPTS[__tamper__] = True

def tamper(payload):
    if False:
        i = 10
        return i + 15

    def add_dollar_at_signs(payload):
        if False:
            while True:
                i = 10
        settings.TAMPER_SCRIPTS[__tamper__] = True
        obf_char = '$@'
        payload = re.sub('([b-zD-Z])', lambda x: obf_char + x[0], payload)
        for word in settings.IGNORE_TAMPER_TRANSFORMATION:
            _ = obf_char.join((word[i:i + 1] for i in range(-1, len(word), 1)))
            if _ in payload:
                payload = payload.replace(_, _.replace(obf_char, ''))
        return payload
    if settings.TARGET_OS != settings.OS.WINDOWS:
        if settings.EVAL_BASED_STATE != False:
            return payload
        else:
            return add_dollar_at_signs(payload)
    else:
        return payload