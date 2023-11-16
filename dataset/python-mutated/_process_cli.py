"""cli-specific implementation of process utilities.

cli - Common Language Infrastructure for IronPython. Code
      can run on any operating system. Check os.name for os-
      specific settings.

This file is only meant to be imported by process.py, not by end-users.

This file is largely untested. To become a full drop-in process
interface for IronPython will probably require you to help fill
in the details. 
"""
import clr
import System
import os
from ._process_common import arg_split

def system(cmd):
    if False:
        for i in range(10):
            print('nop')
    '\n    system(cmd) should work in a cli environment on Mac OSX, Linux,\n    and Windows\n    '
    psi = System.Diagnostics.ProcessStartInfo(cmd)
    psi.RedirectStandardOutput = True
    psi.RedirectStandardError = True
    psi.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal
    psi.UseShellExecute = False
    reg = System.Diagnostics.Process.Start(psi)

def getoutput(cmd):
    if False:
        return 10
    '\n    getoutput(cmd) should work in a cli environment on Mac OSX, Linux,\n    and Windows\n    '
    psi = System.Diagnostics.ProcessStartInfo(cmd)
    psi.RedirectStandardOutput = True
    psi.RedirectStandardError = True
    psi.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal
    psi.UseShellExecute = False
    reg = System.Diagnostics.Process.Start(psi)
    myOutput = reg.StandardOutput
    output = myOutput.ReadToEnd()
    myError = reg.StandardError
    error = myError.ReadToEnd()
    return output

def check_pid(pid):
    if False:
        while True:
            i = 10
    '\n    Check if a process with the given PID (pid) exists\n    '
    try:
        System.Diagnostics.Process.GetProcessById(pid)
        return True
    except System.InvalidOperationException:
        return True
    except System.ArgumentException:
        return False