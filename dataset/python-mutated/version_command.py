"""Version command."""
from __future__ import annotations
import airflow

def version(args):
    if False:
        i = 10
        return i + 15
    'Display Airflow version at the command line.'
    print(airflow.__version__)