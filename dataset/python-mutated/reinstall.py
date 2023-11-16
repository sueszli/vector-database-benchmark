from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from airflow_breeze import NAME
from airflow_breeze.utils.console import get_console

def reinstall_breeze(breeze_sources: Path, re_run: bool=True):
    if False:
        return 10
    '\n    Reinstalls Breeze from specified sources.\n    :param breeze_sources: Sources where to install Breeze from.\n    :param re_run: whether to re-run the original command that breeze was run with.\n    '
    get_console().print(f'\n[info]Reinstalling Breeze from {breeze_sources}\n')
    subprocess.check_call(['pipx', 'install', '-e', str(breeze_sources), '--force'])
    if re_run:
        os.environ['SKIP_UPGRADE_CHECK'] = '1'
        os.execl(sys.executable, sys.executable, *sys.argv)
    get_console().print(f'\n[info]Breeze has been reinstalled from {breeze_sources}. Exiting now.[/]\n\n')
    sys.exit(0)

def warn_non_editable():
    if False:
        while True:
            i = 10
    get_console().print(f'\n[error]Breeze is installed in a wrong way.[/]\n\n[error]It should only be installed in editable mode[/]\n\n[info]Please go to Airflow sources and run[/]\n\n     {NAME} setup self-upgrade --use-current-airflow-sources\n[warning]If during installation, you see warning with "Ignoring --editable install",[/]\n[warning]make sure you first downgrade "packaging" package to <23.2, for example by:[/]\n\n     pip install "packaging<23.2"\n\n')

def warn_dependencies_changed():
    if False:
        while True:
            i = 10
    get_console().print(f'\n[warning]Breeze dependencies changed since the installation![/]\n\n[warning]This might cause various problems!![/]\n\nIf you experience problems - reinstall Breeze with:\n\n    {NAME} setup self-upgrade\n\nThis should usually take couple of seconds.\n')