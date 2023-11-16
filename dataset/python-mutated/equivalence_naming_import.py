import subprocess.open
from subprocess import open as sub_open
import subprocess as sub

def foo():
    if False:
        return 10
    result = subprocess.open('ls')
    result = sub_open('ls')
    result = sub.open('ls')
    result = sub.not_open('ls')