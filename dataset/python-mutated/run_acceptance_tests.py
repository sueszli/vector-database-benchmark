"""Python execution for running acceptance tests."""
from __future__ import annotations
import argparse
import contextlib
import os
import subprocess
import sys
from core.constants import constants
from scripts import build
from scripts import common
from scripts import servers
from typing import Final, List, Optional, Tuple
_PARSER: Final = argparse.ArgumentParser(description="\nRun this script from the oppia root folder:\n   python -m scripts.run_acceptance_tests\n\nThe root folder MUST be named 'oppia'.\n")
_PARSER.add_argument('--skip-build', help='If true, skips building files. The default value is false.', action='store_true')
_PARSER.add_argument('--prod_env', help='Run the tests in prod mode. Static resources are served from build directory and use cache slugs.', action='store_true')
_PARSER.add_argument('--suite', required=True, help='Specifies the test suite to run. For performing a full test, no argument is required.')
_PARSER.add_argument('--server_log_level', help='Sets the log level for the appengine server. The default value is set to error.', default='error', choices=['critical', 'error', 'warning', 'info'])
_PARSER.add_argument('--source_maps', help='Build webpack with source maps.', action='store_true')

def run_tests(args: argparse.Namespace) -> Tuple[List[bytes], int]:
    if False:
        while True:
            i = 10
    'Run the scripts to start acceptance tests.'
    if common.is_oppia_server_already_running():
        sys.exit('\n            Oppia server is already running. Try shutting all the servers down\n            before running the script.\n        ')
    with contextlib.ExitStack() as stack:
        dev_mode = not args.prod_env
        if args.skip_build:
            common.modify_constants(prod_env=args.prod_env)
        else:
            build.build_js_files(dev_mode, source_maps=args.source_maps)
        stack.callback(common.set_constants_to_default)
        stack.enter_context(servers.managed_redis_server())
        stack.enter_context(servers.managed_elasticsearch_dev_server())
        if constants.EMULATOR_MODE:
            stack.enter_context(servers.managed_firebase_auth_emulator())
            stack.enter_context(servers.managed_cloud_datastore_emulator(clear_datastore=True))
        app_yaml_path = 'app.yaml' if args.prod_env else 'app_dev.yaml'
        stack.enter_context(servers.managed_dev_appserver(app_yaml_path, port=common.GAE_PORT_FOR_E2E_TESTING, log_level=args.server_log_level, automatic_restart=False, skip_sdk_update_check=True, env={**os.environ, 'PORTSERVER_ADDRESS': common.PORTSERVER_SOCKET_FILEPATH, 'PIP_NO_DEPS': 'True'}))
        proc = stack.enter_context(servers.managed_acceptance_tests_server(suite_name=args.suite, stdout=subprocess.PIPE))
        print('Servers have come up.\n')
        output_lines = []
        while True:
            for line in iter(proc.stdout.readline, b''):
                if isinstance(line, str):
                    line = line.encode('utf-8')
                output_lines.append(line.rstrip())
                common.write_stdout_safe(line.decode('ascii', errors='replace'))
            if proc.poll() is not None:
                break
        return_value = (output_lines, proc.returncode)
    return return_value

def main(args: Optional[List[str]]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Run acceptance tests.'
    parsed_args = _PARSER.parse_args(args=args)
    with servers.managed_portserver():
        (_, return_code) = run_tests(parsed_args)
    sys.exit(return_code)
if __name__ == '__main__':
    main()