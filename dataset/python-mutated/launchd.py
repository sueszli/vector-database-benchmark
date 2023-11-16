"""
Manage launchd plist files
"""
import os
import sys

def write_launchd_plist(program):
    if False:
        while True:
            i = 10
    '\n    Write a launchd plist for managing salt-master or salt-minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run launchd.write_launchd_plist salt-master\n    '
    plist_sample_text = '\n<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0">\n  <dict>\n    <key>Label</key>\n    <string>org.saltstack.{program}</string>\n    <key>RunAtLoad</key>\n    <true/>\n    <key>KeepAlive</key>\n    <true/>\n    <key>ProgramArguments</key>\n    <array>\n        <string>{script}</string>\n    </array>\n    <key>SoftResourceLimits</key>\n    <dict>\n        <key>NumberOfFiles</key>\n        <integer>100000</integer>\n    </dict>\n    <key>HardResourceLimits</key>\n    <dict>\n        <key>NumberOfFiles</key>\n        <integer>100000</integer>\n    </dict>\n  </dict>\n</plist>\n    '.strip()
    supported_programs = ['salt-master', 'salt-minion']
    if program not in supported_programs:
        sys.stderr.write("Supported programs: '{}'\n".format(supported_programs))
        sys.exit(-1)
        return plist_sample_text.format(program=program, python=sys.executable, script=os.path.join(os.path.dirname(sys.executable), program))