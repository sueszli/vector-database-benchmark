import logging
import site
import sys
from pathlib import Path
from typing import Optional, Tuple
import userpath
from pipx import constants
from pipx.constants import EXIT_CODE_OK, ExitCode
from pipx.emojis import hazard, stars
from pipx.util import pipx_wrap
logger = logging.getLogger(__name__)

def get_pipx_user_bin_path() -> Optional[Path]:
    if False:
        print('Hello World!')
    'Returns None if pipx is not installed using `pip --user`\n    Otherwise returns parent dir of pipx binary\n    '
    pipx_bin_path = None
    script_path = Path(__file__).resolve()
    userbase_path = Path(site.getuserbase()).resolve()
    try:
        _ = script_path.relative_to(userbase_path)
    except ValueError:
        pip_user_installed = False
    else:
        pip_user_installed = True
    if pip_user_installed:
        test_paths = (userbase_path / 'bin' / 'pipx', Path(site.getusersitepackages()).resolve().parent / 'Scripts' / 'pipx.exe')
        for test_path in test_paths:
            if test_path.exists():
                pipx_bin_path = test_path.parent
                break
    return pipx_bin_path

def ensure_path(location: Path, *, force: bool) -> Tuple[bool, bool]:
    if False:
        while True:
            i = 10
    "Ensure location is in user's PATH or add it to PATH.\n    Returns True if location was added to PATH\n    "
    location_str = str(location)
    path_added = False
    need_shell_restart = userpath.need_shell_restart(location_str)
    in_current_path = userpath.in_current_path(location_str)
    if force or (not in_current_path and (not need_shell_restart)):
        path_added = userpath.append(location_str, 'pipx')
        if not path_added:
            print(pipx_wrap(f'{hazard}  {location_str} is not added to the PATH environment variable successfully. You may need to add it to PATH manually.', subsequent_indent=' ' * 4))
        else:
            print(pipx_wrap(f'Success! Added {location_str} to the PATH environment variable.', subsequent_indent=' ' * 4))
        need_shell_restart = userpath.need_shell_restart(location_str)
    elif not in_current_path and need_shell_restart:
        print(pipx_wrap(f'\n                {location_str} has been been added to PATH, but you need to\n                open a new terminal or re-login for this PATH change to take\n                effect.\n                ', subsequent_indent=' ' * 4))
    else:
        print(pipx_wrap(f'{location_str} is already in PATH.', subsequent_indent=' ' * 4))
    return (path_added, need_shell_restart)

def ensure_pipx_paths(force: bool) -> ExitCode:
    if False:
        print('Hello World!')
    'Returns pipx exit code.'
    bin_paths = {constants.LOCAL_BIN_DIR}
    pipx_user_bin_path = get_pipx_user_bin_path()
    if pipx_user_bin_path is not None:
        bin_paths.add(pipx_user_bin_path)
    path_added = False
    need_shell_restart = False
    for bin_path in bin_paths:
        (path_added_current, need_shell_restart_current) = ensure_path(bin_path, force=force)
        path_added |= path_added_current
        need_shell_restart |= need_shell_restart_current
    print()
    if path_added:
        print(pipx_wrap("\n                Consider adding shell completions for pipx. Run 'pipx\n                completions' for instructions.\n                ") + '\n')
    elif not need_shell_restart:
        sys.stdout.flush()
        logger.warning(pipx_wrap(f"\n                {hazard}  All pipx binary directories have been added to PATH. If you\n                are sure you want to proceed, try again with the '--force'\n                flag.\n                ") + '\n')
    if need_shell_restart:
        print(pipx_wrap('\n                You will need to open a new terminal or re-login for the PATH\n                changes to take effect.\n                ') + '\n')
    print(f'Otherwise pipx is ready to go! {stars}')
    return EXIT_CODE_OK