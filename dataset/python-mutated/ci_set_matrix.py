"""
This script is used in GitHub Actions to determine what docs/boards are
built based on what files were changed. The base commit varies depending
on the event that triggered run. Pull request runs will compare to the
base branch while pushes will compare to the current ref. We override this
for the adafruit/circuitpython repo so we build all docs/boards for pushes.

When making changes to the script it is useful to manually test it.
You can for instance run
```shell
tools/ci_set_matrix ports/raspberrypi/common-hal/socket/SSLSocket.c
```
and (at the time this comment was written) get a series of messages indicating
that only the single board raspberry_pi_pico_w would be built.
"""
import re
import os
import sys
import json
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
tools_dir = pathlib.Path(__file__).resolve().parent
top_dir = tools_dir.parent
sys.path.insert(0, str(tools_dir / 'adabot'))
sys.path.insert(0, str(top_dir / 'docs'))
import build_board_info
from shared_bindings_matrix import get_settings_from_makefile, SUPPORTED_PORTS, all_ports_all_boards
IGNORE_BOARD = {'.devcontainer', 'conf.py', 'docs', 'tests', 'tools/ci_changes_per_commit.py', 'tools/ci_check_duplicate_usb_vid_pid.py', 'tools/ci_set_matrix.py'}
PATTERN_DOCS = '^(?:\\.github|docs|extmod\\/ulab)|^(?:(?:ports\\/\\w+\\/bindings|shared-bindings)\\S+\\.c|tools\\/extract_pyi\\.py|\\.readthedocs\\.yml|conf\\.py|requirements-doc\\.txt)$|(?:-stubs|\\.(?:md|MD|mk|rst|RST)|/Makefile)$'
PATTERN_WINDOWS = {'.github/', 'extmod/', 'lib/', 'mpy-cross/', 'ports/unix/', 'py/', 'tools/', 'requirements-dev.txt'}

def git_diff(pattern: str):
    if False:
        while True:
            i = 10
    return set(subprocess.run(f'git diff {pattern} --name-only', capture_output=True, shell=True).stdout.decode('utf-8').split('\n')[:-1])
compute_diff = bool(os.environ.get('BASE_SHA') and os.environ.get('HEAD_SHA'))
if len(sys.argv) > 1:
    print('Using files list on commandline')
    changed_files = set(sys.argv[1:])
elif compute_diff:
    print('Using files list by computing diff')
    changed_files = git_diff('$BASE_SHA...$HEAD_SHA')
    if os.environ.get('GITHUB_EVENT_NAME') == 'pull_request':
        changed_files.intersection_update(git_diff('$GITHUB_SHA~...$GITHUB_SHA'))
else:
    print('Using files list in CHANGED_FILES')
    changed_files = set(json.loads(os.environ.get('CHANGED_FILES') or '[]'))
print('Using jobs list in LAST_FAILED_JOBS')
last_failed_jobs = json.loads(os.environ.get('LAST_FAILED_JOBS') or '{}')

def print_enclosed(title, content):
    if False:
        print('Hello World!')
    print('::group::' + title)
    print(content)
    print('::endgroup::')
print_enclosed('Log: changed_files', changed_files)
print_enclosed('Log: last_failed_jobs', last_failed_jobs)

def set_output(name: str, value):
    if False:
        i = 10
        return i + 15
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'at') as f:
            print(f'{name}={value}', file=f)
    else:
        print(f"Would set GitHub actions output {name} to '{value}'")

def set_boards(build_all: bool):
    if False:
        return 10
    all_board_ids = set()
    boards_to_build = all_board_ids if build_all else set()
    board_to_port = {}
    port_to_board = {}
    board_setting = {}
    for (id, info) in build_board_info.get_board_mapping().items():
        if info.get('alias'):
            continue
        port = info['port']
        all_board_ids.add(id)
        board_to_port[id] = port
        port_to_board.setdefault(port, set()).add(id)

    def compute_board_settings(boards):
        if False:
            while True:
                i = 10
        need = set(boards) - set(board_setting.keys())
        if not need:
            return

        def get_settings(board):
            if False:
                while True:
                    i = 10
            return (board, get_settings_from_makefile(str(top_dir / 'ports' / board_to_port[board]), board))
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            board_setting.update(ex.map(get_settings, need))
    if not build_all:
        pattern_port = re.compile('^ports/([^/]+)/')
        pattern_board = re.compile('^ports/[^/]+/boards/([^/]+)/')
        pattern_module = re.compile('^(ports/[^/]+/(?:common-hal|bindings)|shared-bindings|shared-module)/([^/]+)/')
        for file in changed_files:
            if len(all_board_ids) == len(boards_to_build):
                break
            if any([file.startswith(path) for path in IGNORE_BOARD]):
                continue
            board_matches = pattern_board.search(file)
            if board_matches:
                boards_to_build.add(board_matches.group(1))
                continue
            port_matches = pattern_port.search(file)
            module_matches = pattern_module.search(file)
            port = port_matches.group(1) if port_matches else None
            if port and (not module_matches):
                if port != 'unix':
                    boards_to_build.update(port_to_board[port])
                continue
            if file.startswith('frozen') or file.startswith('supervisor') or module_matches:
                boards = port_to_board[port] if port else all_board_ids
                compute_board_settings(boards)
                for board in boards:
                    settings = board_setting[board]
                    if file.startswith('frozen'):
                        if file in settings['FROZEN_MPY_DIRS']:
                            boards_to_build.add(board)
                            continue
                    if file.startswith('supervisor'):
                        if file in settings['SRC_SUPERVISOR']:
                            boards_to_build.add(board)
                            continue
                        if file.startswith('supervisor/shared/web_workflow/static/'):
                            web_workflow = settings['CIRCUITPY_WEB_WORKFLOW']
                            if web_workflow != '0':
                                boards_to_build.add(board)
                                continue
                    if module_matches:
                        module = module_matches.group(2) + '/'
                        if module in settings['SRC_PATTERNS']:
                            boards_to_build.add(board)
                            continue
                continue
            boards_to_build = all_board_ids
            break
    boards_to_build.update(last_failed_jobs.get('ports', []))
    print('Building boards:', bool(boards_to_build))
    port_to_boards_to_build = {}
    for board in sorted(boards_to_build):
        port = board_to_port.get(board)
        if not port:
            continue
        port_to_boards_to_build.setdefault(port, []).append(board)
        print(' ', board)
    if port_to_boards_to_build:
        port_to_boards_to_build['ports'] = sorted(list(port_to_boards_to_build.keys()))
    set_output('ports', json.dumps(port_to_boards_to_build))

def set_docs(run: bool):
    if False:
        for i in range(10):
            print('nop')
    if not run:
        if last_failed_jobs.get('docs'):
            run = True
        else:
            pattern_doc = re.compile(PATTERN_DOCS)
            github_workspace = os.environ.get('GITHUB_WORKSPACE') or ''
            github_workspace = github_workspace and github_workspace + '/'
            for file in changed_files:
                if pattern_doc.search(file) and (subprocess.run(f"git diff -U0 $BASE_SHA...$HEAD_SHA {github_workspace + file} | grep -o -m 1 '^[+-]\\/\\/|'", capture_output=True, shell=True).stdout if file.endswith('.c') else True):
                    run = True
                    break
    print('Building docs:', run)
    set_output('docs', run)

def set_windows(run: bool):
    if False:
        print('Hello World!')
    if not run:
        if last_failed_jobs.get('windows'):
            run = True
        else:
            for file in changed_files:
                for pattern in PATTERN_WINDOWS:
                    if file.startswith(pattern) and (not any([file.startswith(path) for path in IGNORE_BOARD])):
                        run = True
                        break
                else:
                    continue
                break
    print('Building windows:', run)
    set_output('windows', run)

def main():
    if False:
        print('Hello World!')
    run_all = not changed_files and (not compute_diff)
    print('Running: ' + ('all' if run_all else 'conditionally'))
    set_docs(run_all)
    set_windows(run_all)
    set_boards(run_all)
if __name__ == '__main__':
    main()