"""
This module provides the logic for the `pyre restart` command, which
is a single command that effectively runs `pyre stop` followed by
`pyre incremental`.
"""
import logging
from .. import command_arguments, frontend_configuration, identifiers
from . import commands, incremental, stop
LOG: logging.Logger = logging.getLogger(__name__)

def run(configuration: frontend_configuration.Base, incremental_arguments: command_arguments.IncrementalArguments) -> commands.ExitCode:
    if False:
        i = 10
        return i + 15
    stop.run(configuration, flavor=identifiers.PyreFlavor.CLASSIC)
    return incremental.run_incremental(configuration, incremental_arguments).exit_code