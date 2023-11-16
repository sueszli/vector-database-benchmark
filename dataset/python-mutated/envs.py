"""
Python environments general utilities
"""
from spyder.utils.conda import get_list_conda_envs
from spyder.utils.pyenv import get_list_pyenv_envs

def get_list_envs():
    if False:
        while True:
            i = 10
    '\n    Get the list of environments in the system.\n\n    Currently detected conda and pyenv based environments.\n    '
    conda_env = get_list_conda_envs()
    pyenv_env = get_list_pyenv_envs()
    return {**conda_env, **pyenv_env}