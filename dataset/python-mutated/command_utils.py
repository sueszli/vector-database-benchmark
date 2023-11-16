from __future__ import annotations
import shlex
import subprocess

def run_command(cmd: list[str], *, print_output_on_error: bool=True, return_output: bool=False, check=True, **kwargs) -> str | bool:
    if False:
        while True:
            i = 10
    print(f"Running command: {' '.join((shlex.quote(c) for c in cmd))}")
    try:
        if return_output:
            return subprocess.check_output(cmd, **kwargs).decode()
        else:
            try:
                result = subprocess.run(cmd, check=check, **kwargs)
                return result.returncode == 0
            except FileNotFoundError:
                if check:
                    raise
                else:
                    return False
    except subprocess.CalledProcessError as ex:
        if print_output_on_error:
            print('========================= OUTPUT start ============================')
            print(ex.stderr)
            print(ex.stdout)
            print('========================= OUTPUT end ============================')
        raise