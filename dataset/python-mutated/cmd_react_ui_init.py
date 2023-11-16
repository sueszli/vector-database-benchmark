import os
import re
import shutil
import subprocess
from typing import Optional
from lightning.app.utilities.app_helpers import Logger
logger = Logger(__name__)

def react_ui(dest_dir: Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    _check_react_prerequisites()
    _copy_and_setup_react_ui(dest_dir)

def _copy_and_setup_react_ui(dest_dir: Optional[str]=None) -> None:
    if False:
        return 10
    logger.info('⚡ setting up react-ui template')
    path = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(path, 'react-ui-template')
    if dest_dir is None:
        dest_dir = os.path.join(os.getcwd(), 'react-ui')
    shutil.copytree(template_dir, dest_dir)
    logger.info('⚡ install react project deps')
    ui_path = os.path.join(dest_dir, 'ui')
    subprocess.run(f'cd {ui_path} && yarn install', shell=True)
    logger.info('⚡ building react project')
    subprocess.run(f'cd {ui_path} && yarn build', shell=True)
    m = f'\n    ⚡⚡ react-ui created! ⚡⚡\n\n    ⚡ Connect it to your component using `configure_layout`:\n\n    # Use a LightningFlow or LightningWork\n    class YourComponent(la.LightningFlow):\n        def configure_layout(self):\n            return la.frontend.StaticWebFrontend(Path(__file__).parent / "react-ui/src/dist")\n\n    ⚡ run the example_app.py to see it live!\n    lightning run app {dest_dir}/example_app.py\n\n    '
    logger.info(m)

def _check_react_prerequisites() -> None:
    if False:
        return 10
    'Args are for test purposes only.'
    missing_msgs = []
    version_regex = '\\d{1,2}\\.\\d{1,2}\\.\\d{1,3}'
    logger.info('Checking pre-requisites for react')
    npm_version = subprocess.check_output(['npm', '--version'])
    has_npm = bool(re.search(version_regex, str(npm_version)))
    npm_version = re.search(version_regex, str(npm_version))
    npm_version = None if npm_version is None else npm_version.group(0)
    if not has_npm:
        m = "\n        This machine is missing 'npm'. Please install npm and rerun 'lightning init react-ui' again.\n\n        Install instructions: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm\n        "
        missing_msgs.append(m)
    node_version = subprocess.check_output(['node', '--version'])
    has_node = bool(re.search(version_regex, str(node_version)))
    node_version = re.search(version_regex, str(node_version))
    node_version = None if node_version is None else node_version.group(0)
    if not has_node:
        m = "\n        This machine is missing 'node'. Please install node and rerun 'lightning init react-ui' again.\n\n        Install instructions: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm\n        "
        missing_msgs.append(m)
    yarn_version = subprocess.check_output(['yarn', '--version'])
    has_yarn = bool(re.search(version_regex, str(yarn_version)))
    yarn_version = re.search(version_regex, str(yarn_version))
    yarn_version = None if yarn_version is None else yarn_version.group(0)
    if not has_yarn:
        m = "\n        This machine is missing 'yarn'. Please install npm+node first, then run\n\n        npm install --global yarn\n\n        Full install instructions: https://classic.yarnpkg.com/lang/en/docs/install/#mac-stable\n        "
        missing_msgs.append(m)
    if len(missing_msgs) > 0:
        missing_msg = '\n'.join(missing_msgs)
        raise SystemExit(missing_msg)
    logger.info(f'\n    found npm  version: {npm_version}\n    found node version: {node_version}\n    found yarn version: {yarn_version}\n\n    Pre-requisites met!\n    ')