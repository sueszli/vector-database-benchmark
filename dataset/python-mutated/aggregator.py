"""Aggregation function for CLI specified options and config file options.

This holds the logic that uses the collected and merged config files and
applies the user-specified command-line configuration on top of it.
"""
from __future__ import annotations
import argparse
import configparser
import logging
from typing import Sequence
from flake8.options import config
from flake8.options.manager import OptionManager
LOG = logging.getLogger(__name__)

def aggregate_options(manager: OptionManager, cfg: configparser.RawConfigParser, cfg_dir: str, argv: Sequence[str] | None) -> argparse.Namespace:
    if False:
        for i in range(10):
            print('nop')
    'Aggregate and merge CLI and config file options.'
    default_values = manager.parse_args([])
    parsed_config = config.parse_config(manager, cfg, cfg_dir)
    default_values.extended_default_ignore = manager.extended_default_ignore
    default_values.extended_default_select = manager.extended_default_select
    for (config_name, value) in parsed_config.items():
        dest_name = config_name
        if not hasattr(default_values, config_name):
            dest_val = manager.config_options_dict[config_name].dest
            assert isinstance(dest_val, str)
            dest_name = dest_val
        LOG.debug('Overriding default value of (%s) for "%s" with (%s)', getattr(default_values, dest_name, None), dest_name, value)
        setattr(default_values, dest_name, value)
    return manager.parse_args(argv, default_values)