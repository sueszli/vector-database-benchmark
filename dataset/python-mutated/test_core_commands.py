"""Tests for the core commands implemented in git-sim.

All test runs use the -d flag to prevent images from opening automatically.

To induce failure, include a call to `run_git_reset()` in one of the
  test functions.
"""
import os, subprocess
from pathlib import Path
from utils import get_cmd_parts, compare_images, run_git_reset
import pytest
git_sim_commands = ['git-sim add', 'git-sim log', 'git-sim clean', 'git-sim commit', 'git-sim restore', 'git-sim stash', 'git-sim status', 'git-sim branch new_branch', 'git-sim checkout branch2', 'git-sim cherry-pick branch2', 'git-sim merge branch2', 'git-sim mv main.1 main.100', 'git-sim rebase branch2', 'git-sim reset HEAD^', 'git-sim revert HEAD^', 'git-sim rm main.1', 'git-sim switch branch2', 'git-sim tag new_tag']

@pytest.mark.parametrize('raw_cmd', git_sim_commands)
def test_command(tmp_repo, raw_cmd):
    if False:
        print('Hello World!')
    'Test a  git-sim command.\n\n    This function works for any command of the forms\n      `git-sim <command`\n      `git-sim <command> <arg>`\n    '
    raw_cmd_parts = raw_cmd.split(' ')
    filename_element = f"git-sim-{raw_cmd_parts[1].replace('-', '_')}"
    cmd_parts = get_cmd_parts(raw_cmd)
    os.chdir(tmp_repo)
    output = subprocess.run(cmd_parts, capture_output=True)
    fp_generated = Path(output.stdout.decode().strip())
    fp_reference = Path(__file__).parent / f'reference_files/{filename_element}.png'
    assert filename_element in str(fp_generated)
    compare_images(fp_generated, fp_reference)