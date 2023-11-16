from __future__ import annotations
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator
if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config
from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER
from autogpt.models.command import Command
logger = logging.getLogger(__name__)

class CommandRegistry:
    """
    The CommandRegistry class is a manager for a collection of Command objects.
    It allows the registration, modification, and retrieval of Command objects,
    as well as the scanning and loading of command plugins from a specified
    directory.
    """
    commands: dict[str, Command]
    commands_aliases: dict[str, Command]
    categories: dict[str, CommandCategory]

    @dataclass
    class CommandCategory:
        name: str
        title: str
        description: str
        commands: list[Command] = field(default_factory=list[Command])
        modules: list[ModuleType] = field(default_factory=list[ModuleType])

    def __init__(self):
        if False:
            while True:
                i = 10
        self.commands = {}
        self.commands_aliases = {}
        self.categories = {}

    def __contains__(self, command_name: str):
        if False:
            for i in range(10):
                print('nop')
        return command_name in self.commands or command_name in self.commands_aliases

    def _import_module(self, module_name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return importlib.import_module(module_name)

    def _reload_module(self, module: Any) -> Any:
        if False:
            while True:
                i = 10
        return importlib.reload(module)

    def register(self, cmd: Command) -> None:
        if False:
            i = 10
            return i + 15
        if cmd.name in self.commands:
            logger.warn(f"Command '{cmd.name}' already registered and will be overwritten!")
        self.commands[cmd.name] = cmd
        if cmd.name in self.commands_aliases:
            logger.warn(f"Command '{cmd.name}' will overwrite alias with the same name of '{self.commands_aliases[cmd.name]}'!")
        for alias in cmd.aliases:
            self.commands_aliases[alias] = cmd

    def unregister(self, command: Command) -> None:
        if False:
            return 10
        if command.name in self.commands:
            del self.commands[command.name]
            for alias in command.aliases:
                del self.commands_aliases[alias]
        else:
            raise KeyError(f"Command '{command.name}' not found in registry.")

    def reload_commands(self) -> None:
        if False:
            return 10
        'Reloads all loaded command plugins.'
        for cmd_name in self.commands:
            cmd = self.commands[cmd_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, 'register'):
                reloaded_module.register(self)

    def get_command(self, name: str) -> Command | None:
        if False:
            return 10
        if name in self.commands:
            return self.commands[name]
        if name in self.commands_aliases:
            return self.commands_aliases[name]

    def call(self, command_name: str, agent: BaseAgent, **kwargs) -> Any:
        if False:
            print('Hello World!')
        if (command := self.get_command(command_name)):
            return command(**kwargs, agent=agent)
        raise KeyError(f"Command '{command_name}' not found in registry")

    def list_available_commands(self, agent: BaseAgent) -> Iterator[Command]:
        if False:
            print('Hello World!')
        'Iterates over all registered commands and yields those that are available.\n\n        Params:\n            agent (BaseAgent): The agent that the commands will be checked against.\n\n        Yields:\n            Command: The next available command.\n        '
        for cmd in self.commands.values():
            available = cmd.available
            if callable(cmd.available):
                available = cmd.available(agent)
            if available:
                yield cmd

    @staticmethod
    def with_command_modules(modules: list[str], config: Config) -> CommandRegistry:
        if False:
            return 10
        new_registry = CommandRegistry()
        logger.debug(f'The following command categories are disabled: {config.disabled_command_categories}')
        enabled_command_modules = [x for x in modules if x not in config.disabled_command_categories]
        logger.debug(f'The following command categories are enabled: {enabled_command_modules}')
        for command_module in enabled_command_modules:
            new_registry.import_command_module(command_module)
        for command in [c for c in new_registry.commands.values()]:
            if callable(command.enabled) and (not command.enabled(config)):
                new_registry.unregister(command)
                logger.debug(f'''Unregistering incompatible command '{command.name}': "{command.disabled_reason or 'Disabled by current config.'}"''')
        return new_registry

    def import_command_module(self, module_name: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Imports the specified Python module containing command plugins.\n\n        This method imports the associated module and registers any functions or\n        classes that are decorated with the `AUTO_GPT_COMMAND_IDENTIFIER` attribute\n        as `Command` objects. The registered `Command` objects are then added to the\n        `commands` dictionary of the `CommandRegistry` object.\n\n        Args:\n            module_name (str): The name of the module to import for command plugins.\n        '
        module = importlib.import_module(module_name)
        category = self.register_module_category(module)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            command = None
            if getattr(attr, AUTO_GPT_COMMAND_IDENTIFIER, False):
                command = attr.command
            elif inspect.isclass(attr) and issubclass(attr, Command) and (attr != Command):
                command = attr()
            if command:
                self.register(command)
                category.commands.append(command)

    def register_module_category(self, module: ModuleType) -> CommandCategory:
        if False:
            i = 10
            return i + 15
        if not (category_name := getattr(module, 'COMMAND_CATEGORY', None)):
            raise ValueError(f'Cannot import invalid command module {module.__name__}')
        if category_name not in self.categories:
            self.categories[category_name] = CommandRegistry.CommandCategory(name=category_name, title=getattr(module, 'COMMAND_CATEGORY_TITLE', category_name.capitalize()), description=getattr(module, '__doc__', ''))
        category = self.categories[category_name]
        if module not in category.modules:
            category.modules.append(module)
        return category