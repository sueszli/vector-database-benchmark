"""This script performs lighthouse checks and creates lighthouse reports.
Any callers must pass in a flag, either --accessibility or --performance.
"""
from __future__ import annotations
import argparse
import contextlib
import os
import subprocess
import sys
from typing import Final, List, Optional
from scripts import common
from core.constants import constants
from scripts import build
from scripts import servers
LIGHTHOUSE_MODE_PERFORMANCE: Final = 'performance'
LIGHTHOUSE_MODE_ACCESSIBILITY: Final = 'accessibility'
SERVER_MODE_PROD: Final = 'dev'
SERVER_MODE_DEV: Final = 'prod'
GOOGLE_APP_ENGINE_PORT: Final = 8181
LIGHTHOUSE_CONFIG_FILENAMES: Final = {LIGHTHOUSE_MODE_PERFORMANCE: {'1': '.lighthouserc-1.js', '2': '.lighthouserc-2.js'}, LIGHTHOUSE_MODE_ACCESSIBILITY: {'1': '.lighthouserc-accessibility-1.js', '2': '.lighthouserc-accessibility-2.js'}}
APP_YAML_FILENAMES: Final = {SERVER_MODE_PROD: 'app.yaml', SERVER_MODE_DEV: 'app_dev.yaml'}
_PARSER: Final = argparse.ArgumentParser(description="\nRun the script from the oppia root folder:\n    python -m scripts.run_lighthouse_tests\nNote that the root folder MUST be named 'oppia'.\n")
_PARSER.add_argument('--mode', help='Sets the mode for the lighthouse tests', required=True, choices=['accessibility', 'performance'])
_PARSER.add_argument('--shard', help='Sets the shard for the lighthouse tests', required=True, choices=['1', '2'])
_PARSER.add_argument('--skip_build', help='Sets whether to skip webpack build', action='store_true')
_PARSER.add_argument('--record_screen', help='Sets whether LHCI Puppeteer script is recorded', action='store_true')

def run_lighthouse_puppeteer_script(record: bool=False) -> None:
    if False:
        print('Hello World!')
    'Runs puppeteer script to collect dynamic urls.\n\n    Args:\n        record: bool. Set to True to record the LHCI puppeteer script\n            via puppeteer-screen-recorder and False to not. Note that\n            puppeteer-screen-recorder must be separately installed to record.\n    '
    puppeteer_path = os.path.join('core', 'tests', 'puppeteer', 'lighthouse_setup.js')
    bash_command = [common.NODE_BIN_PATH, puppeteer_path]
    if record:
        bash_command.append('-record')
        dir_path = os.path.join(os.getcwd(), '..', 'lhci-puppeteer-video')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        video_path = os.path.join(dir_path, 'video.mp4')
        bash_command.append(video_path)
        print('Starting LHCI Puppeteer script with recording.')
        print('Video Path:' + video_path)
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = process.communicate()
    if process.returncode == 0:
        print(stdout)
        for line in stdout.split(b'\n'):
            export_url(line.decode('utf-8'))
        print('Puppeteer script completed successfully.')
        if record:
            print('Resulting puppeteer video saved at %s' % video_path)
    else:
        print('Return code: %s' % process.returncode)
        print('OUTPUT:')
        print(stdout.decode('utf-8'))
        print('ERROR:')
        print(stderr.decode('utf-8'))
        print('Puppeteer script failed. More details can be found above.')
        if record:
            print('Resulting puppeteer video saved at %s' % video_path)
        sys.exit(1)

def run_webpack_compilation() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Runs webpack compilation.'
    max_tries = 5
    webpack_bundles_dir_name = 'webpack_bundles'
    for _ in range(max_tries):
        try:
            with servers.managed_webpack_compiler() as proc:
                proc.wait()
        except subprocess.CalledProcessError as error:
            print(error.output)
            sys.exit(error.returncode)
        if os.path.isdir(webpack_bundles_dir_name):
            break
    if not os.path.isdir(webpack_bundles_dir_name):
        print('Failed to complete webpack compilation, exiting...')
        sys.exit(1)

def export_url(line: str) -> None:
    if False:
        return 10
    'Exports the entity ID in the given line to an environment variable, if\n    the line is a URL.\n\n    Args:\n        line: str. The line to parse and extract the entity ID from. If no\n            recognizable URL is present, nothing is exported to the\n            environment.\n    '
    url_parts = line.split('/')
    print('Parsing and exporting entity ID in line: %s' % line)
    if 'create' in line:
        os.environ['exploration_id'] = url_parts[4]
    elif 'topic_editor' in line:
        os.environ['topic_id'] = url_parts[4]
    elif 'story_editor' in line:
        os.environ['story_id'] = url_parts[4]
    elif 'skill_editor' in line:
        os.environ['skill_id'] = url_parts[4]

def run_lighthouse_checks(lighthouse_mode: str, shard: str) -> None:
    if False:
        print('Hello World!')
    'Runs the Lighthouse checks through the Lighthouse config.\n\n    Args:\n        lighthouse_mode: str. Represents whether the lighthouse checks are in\n            accessibility mode or performance mode.\n        shard: str. Specifies which shard of the tests should be run.\n    '
    lhci_path = os.path.join('node_modules', '@lhci', 'cli', 'src', 'cli.js')
    bash_command = [common.NODE_BIN_PATH, lhci_path, 'autorun', '--config=%s' % LIGHTHOUSE_CONFIG_FILENAMES[lighthouse_mode][shard], '--max-old-space-size=4096']
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = process.communicate()
    if process.returncode == 0:
        print('Lighthouse checks completed successfully.')
    else:
        print('Return code: %s' % process.returncode)
        print('OUTPUT:')
        print(stdout.decode('utf-8'))
        print('ERROR:')
        print(stderr.decode('utf-8'))
        print('Lighthouse checks failed. More details can be found above.')
        sys.exit(1)

def main(args: Optional[List[str]]=None) -> None:
    if False:
        while True:
            i = 10
    'Runs lighthouse checks and deletes reports.'
    parsed_args = _PARSER.parse_args(args=args)
    common.setup_chrome_bin_env_variable()
    if parsed_args.mode == LIGHTHOUSE_MODE_ACCESSIBILITY:
        lighthouse_mode = LIGHTHOUSE_MODE_ACCESSIBILITY
        server_mode = SERVER_MODE_DEV
    else:
        lighthouse_mode = LIGHTHOUSE_MODE_PERFORMANCE
        server_mode = SERVER_MODE_PROD
    if lighthouse_mode == LIGHTHOUSE_MODE_PERFORMANCE:
        if not parsed_args.skip_build:
            print('Building files in production mode.')
            build.main(args=['--prod_env'])
        else:
            print('Building files in production mode skipping webpack build.')
            build.main(args=[])
            common.run_ng_compilation()
            run_webpack_compilation()
    else:
        build.main(args=[])
        common.run_ng_compilation()
        run_webpack_compilation()
    with contextlib.ExitStack() as stack:
        stack.enter_context(servers.managed_redis_server())
        stack.enter_context(servers.managed_elasticsearch_dev_server())
        if constants.EMULATOR_MODE:
            stack.enter_context(servers.managed_firebase_auth_emulator())
            stack.enter_context(servers.managed_cloud_datastore_emulator())
        env = os.environ.copy()
        env['PIP_NO_DEPS'] = 'True'
        stack.enter_context(servers.managed_dev_appserver(APP_YAML_FILENAMES[server_mode], port=GOOGLE_APP_ENGINE_PORT, log_level='critical', skip_sdk_update_check=True, env=env))
        run_lighthouse_puppeteer_script(parsed_args.record_screen)
        run_lighthouse_checks(lighthouse_mode, parsed_args.shard)
if __name__ == '__main__':
    main()