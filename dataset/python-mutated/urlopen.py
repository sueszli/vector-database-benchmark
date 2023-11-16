""" Example dangerous usage of urllib.request opener functions

The urllib.request opener functions and object can open http, ftp,
and file urls. Often, the ability to open file urls is overlooked leading
to code that can unexpectedly open files on the local server. This
could be used by an attacker to leak information about the server.
"""
import urllib.request
import six

def test_urlopen():
    if False:
        print('Hello World!')
    urllib.request.urlopen('file:///bin/ls')
    urllib.request.urlretrieve('file:///bin/ls', '/bin/ls2')
    opener = urllib.request.URLopener()
    opener.open('file:///bin/ls')
    opener.retrieve('file:///bin/ls')
    opener = urllib.request.FancyURLopener()
    opener.open('file:///bin/ls')
    opener.retrieve('file:///bin/ls')
    six.moves.urllib.request.urlopen('file:///bin/ls')
    six.moves.urllib.request.urlretrieve('file:///bin/ls', '/bin/ls2')
    opener = six.moves.urllib.request.URLopener()
    opener.open('file:///bin/ls')
    opener.retrieve('file:///bin/ls')
    opener = six.moves.urllib.request.FancyURLopener()
    opener.open('file:///bin/ls')
    opener.retrieve('file:///bin/ls')