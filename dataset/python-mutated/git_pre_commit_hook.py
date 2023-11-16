import json
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from lxml import html
'\nHook to make the commit command automatically close bugs when the commit\nmessage contains `Fix #number` or `Implement #number`. Also updates the commit\nmessage with the summary of the closed bug.\n\n'
LAUNCHPAD = os.path.expanduser('~/work/env/launchpad.py')
LAUNCHPAD_BUG = 'https://bugs.launchpad.net/calibre/+bug/%s'
GITHUB_BUG = 'https://api.github.com/repos/kovidgoyal/calibre/issues/%s'
BUG_PAT = '(Fix|Implement|Fixes|Fixed|Implemented|See)\\s+#(\\d+)'
socket.setdefaulttimeout(90)

class Bug:

    def __init__(self):
        if False:
            print('Hello World!')
        self.seen = set()

    def __call__(self, match):
        if False:
            return 10
        (action, bug) = (match.group(1), match.group(2))
        summary = ''
        if bug in self.seen:
            return match.group()
        self.seen.add(bug)
        if int(bug) > 100000:
            try:
                raw = urllib.request.urlopen(LAUNCHPAD_BUG % bug).read()
                h1 = html.fromstring(raw).xpath('//h1[@id="edit-title"]')[0]
                summary = html.tostring(h1, method='text', encoding=str).strip()
            except:
                summary = 'Private bug'
        else:
            summary = json.loads(urllib.request.urlopen(GITHUB_BUG % bug).read())['title']
        if summary:
            print('Working on bug:', summary)
            if int(bug) > 100000 and action != 'See':
                self.close_bug(bug, action)
                return match.group() + f' [{summary}]({LAUNCHPAD_BUG % bug})'
            return match.group() + ' (%s)' % summary
        return match.group()

    def close_bug(self, bug, action):
        if False:
            return 10
        print('Closing bug #%s' % bug)
        suffix = 'The fix will be in the next release. calibre is usually released every alternate Friday.'
        action += 'ed'
        msg = '{} in branch {}. {}'.format(action, 'master', suffix)
        msg = msg.replace('Fixesed', 'Fixed')
        env = dict(os.environ)
        env['LAUNCHPAD_FIX_BUG'] = msg
        subprocess.run([sys.executable, LAUNCHPAD], env=env, input=f'Subject: [Bug {bug}]', text=True, check=True)

def main():
    if False:
        i = 10
        return i + 15
    with open(sys.argv[-1], 'r+b') as f:
        raw = f.read().decode('utf-8')
        bug = Bug()
        msg = re.sub(BUG_PAT, bug, raw)
        if msg != raw:
            f.seek(0)
            f.truncate()
            f.write(msg.encode('utf-8'))
if __name__ == '__main__':
    main()