import logging
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def activate(venv_path: Path) -> dict:
    if False:
        i = 10
        return i + 15
    import copy
    oldenv = copy.deepcopy(os.environ)
    os.environ['VIRTUAL_ENV'] = str(venv_path.absolute())
    os.environ['PATH'] = os.pathsep.join([str(venv_path.absolute() / 'bin'), os.environ['PATH']])
    os.environ['PYTHONPATH'] = str(venv_path.absolute() / 'lib')
    return oldenv

def deactivate(oldenv: dict):
    if False:
        i = 10
        return i + 15
    os.environ = oldenv

def run_semgrep_version(venv_builder: Optional[venv.EnvBuilder]=None, semgrep_version: str='local', semgrep_local_path: Optional[Path]=None, script_arguments: Optional[list]=None):
    if False:
        i = 10
        return i + 15
    if semgrep_version == 'local' and (not semgrep_local_path):
        raise ValueError(f"Parameter 'path' needs a value when run with version='local'. 'path' is currently '{semgrep_local_path}'")
    if not script_arguments:
        script_arguments = None
    venv_path = Path(f'semgrep-{semgrep_version}')
    logger.info(f'Creating virtual environment at {str(venv_path.absolute())}')
    venv_builder.create(venv_path.absolute())
    oldenv = activate(venv_path.absolute())
    if semgrep_version == 'local':
        logger.info(f'Installing semgrep from {semgrep_local_path.absolute()}')
        subprocess.run(['pip', 'install', semgrep_local_path])
    elif semgrep_version == 'latest':
        logger.info(f'Installing latest semgrep')
        subprocess.run(['pip', 'install', '--upgrade', 'semgrep'])
    else:
        logger.info(f'Installing semgrep=={semgrep_version}')
        subprocess.run(['pip', 'install', f'semgrep=={semgrep_version}'])
    r = subprocess.run(['semgrep', '--version'], stdout=subprocess.PIPE)
    real_semgrep_version = r.stdout.decode('utf-8')
    logger.info(f'Using semgrep version {real_semgrep_version}')
    subprocess.run(['./run-benchmarks'] + script_arguments)
    deactivate(oldenv)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--against', '-a', default='latest')
    parser.add_argument('--local', '-l', required=True)
    (args, remaining) = parser.parse_known_args()
    builder = venv.EnvBuilder(clear=True, with_pip=True)
    run_semgrep_version(venv_builder=builder, semgrep_version=args.against, script_arguments=remaining)
    run_semgrep_version(venv_builder=builder, semgrep_version='local', semgrep_local_path=Path(args.local), script_arguments=remaining)