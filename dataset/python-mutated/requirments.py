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
import subprocess
'\nCheck for requirments.\n'

def do_check(requirment):
    if False:
        i = 10
        return i + 15
    try:
        null = open(os.devnull, 'w')
        subprocess.Popen(requirment, stdout=null, stderr=null)
        null.close()
        return True
    except OSError:
        return False