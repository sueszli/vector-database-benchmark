"""Logic for checking and displaying man pages."""
import subprocess
import os
from httpie.context import Environment
MAN_COMMAND = 'man'
NO_MAN_PAGES = os.getenv('HTTPIE_NO_MAN_PAGES', False)
MAN_PAGE_SECTION = '1'

def is_available(program: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "\n    Check whether `program`'s man pages are available on this system.\n\n    "
    if NO_MAN_PAGES or os.system == 'nt':
        return False
    try:
        process = subprocess.run([MAN_COMMAND, MAN_PAGE_SECTION, program], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    else:
        return process.returncode == 0

def display_for(env: Environment, program: str) -> None:
    if False:
        return 10
    '\n    Open the system man page for the given command (http/https/httpie).\n\n    '
    subprocess.run([MAN_COMMAND, MAN_PAGE_SECTION, program], stdout=env.stdout, stderr=env.stderr)