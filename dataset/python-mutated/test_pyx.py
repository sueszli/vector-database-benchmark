"""
External link to pyx
"""
import os
import subprocess
from scapy.error import log_loading
__all__ = ['PYX']

def _test_pyx():
    if False:
        i = 10
        return i + 15
    'Returns if PyX is correctly installed or not'
    try:
        with open(os.devnull, 'wb') as devnull:
            r = subprocess.check_call(['pdflatex', '--version'], stdout=devnull, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, OSError):
        return False
    else:
        return r == 0
try:
    import pyx
    if _test_pyx():
        PYX = 1
    else:
        log_loading.info('PyX dependencies are not installed ! Please install TexLive or MikTeX.')
        PYX = 0
except ImportError:
    log_loading.info("Can't import PyX. Won't be able to use psdump() or pdfdump().")
    PYX = 0