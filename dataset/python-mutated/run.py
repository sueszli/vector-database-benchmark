from __future__ import annotations
from typing import TYPE_CHECKING
from cleo.helpers import argument
from poetry.console.commands.env_command import EnvCommand
from poetry.utils._compat import WINDOWS
if TYPE_CHECKING:
    from poetry.core.masonry.utils.module import Module

class RunCommand(EnvCommand):
    name = 'run'
    description = 'Runs a command in the appropriate environment.'
    arguments = [argument('args', 'The command and arguments/options to run.', multiple=True)]

    def handle(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        args = self.argument('args')
        script = args[0]
        scripts = self.poetry.local_config.get('scripts')
        if scripts and script in scripts:
            return self.run_script(scripts[script], args)
        try:
            return self.env.execute(*args)
        except FileNotFoundError:
            self.line_error(f'<error>Command not found: <c1>{script}</c1></error>')
            return 1

    @property
    def _module(self) -> Module:
        if False:
            while True:
                i = 10
        from poetry.core.masonry.utils.module import Module
        poetry = self.poetry
        package = poetry.package
        path = poetry.file.path.parent
        module = Module(package.name, path.as_posix(), package.packages)
        return module

    def run_script(self, script: str | dict[str, str], args: list[str]) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Runs an entry point script defined in the section ``[tool.poetry.scripts]``.\n\n        When a script exists in the venv bin folder, i.e. after ``poetry install``,\n        then ``sys.argv[0]`` must be set to the full path of the executable, so\n        ``poetry run foo`` and ``poetry shell``, ``foo`` have the same ``sys.argv[0]``\n        that points to the full path.\n\n        Otherwise (when an entry point script does not exist), ``sys.argv[0]`` is the\n        script name only, i.e. ``poetry run foo`` has ``sys.argv == ['foo']``.\n        "
        for script_dir in self.env.script_dirs:
            script_path = script_dir / args[0]
            if WINDOWS:
                script_path = script_path.with_suffix('.cmd')
            if script_path.exists():
                args = [str(script_path), *args[1:]]
                break
        else:
            self._warning_not_installed_script(args[0])
        if isinstance(script, dict):
            script = script['callable']
        (module, callable_) = script.split(':')
        src_in_sys_path = "sys.path.append('src'); " if self._module.is_in_src() else ''
        cmd = ['python', '-c']
        cmd += [f"import sys; from importlib import import_module; sys.argv = {args!r}; {src_in_sys_path}sys.exit(import_module('{module}').{callable_}())"]
        return self.env.execute(*cmd)

    def _warning_not_installed_script(self, script: str) -> None:
        if False:
            return 10
        message = f"Warning: '{script}' is an entry point defined in pyproject.toml, but it's not installed as a script. You may get improper `sys.argv[0]`.\n\nThe support to run uninstalled scripts will be removed in a future release.\n\nRun `poetry install` to resolve and get rid of this message.\n"
        self.line_error(message, style='warning')