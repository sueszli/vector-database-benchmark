import subprocess
import hypothesistooling as tools
from hypothesistooling import installers as install
SCRIPTS = [f for f in tools.all_files() if f.endswith('.sh')]

def test_all_shell_scripts_are_valid():
    if False:
        while True:
            i = 10
    subprocess.check_call([install.SHELLCHECK, '--exclude=SC1073,SC1072', *SCRIPTS], cwd=tools.ROOT)