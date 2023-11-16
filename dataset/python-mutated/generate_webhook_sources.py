from templates.sources.db import db
from templates.sources.webhooks.async_run import async_run
from templates.sources.webhooks.additional_rules import additional_rules
from templates.sources.webhooks.hooks import hooks
from templates.sources.webhooks.http import http
from templates.sources.webhooks.introduction import introduction
from templates.sources.webhooks.main_example import main_example
from templates.sources.webhooks.other_example import other_example
from templates.common.app_prop import app_prop
from templates.common.auth import auth
from templates.common.common_files import common_files
from templates.common.component_metadata import source_metadata
from templates.common.platform_axios import platform_axios
from templates.common.props import props
from templates.common.rules import rules
from templates.common.async_options import async_options
from templates.common.typescript_definitions import typescript_definitions
from templates.common.end import end
checks = [app_prop, auth, props, async_run, hooks, http, platform_axios, async_options, source_metadata, rules, additional_rules, typescript_definitions, end]
always_include = [introduction, typescript_definitions, main_example, other_example, end]

def system_instructions(auth_details='', parsed_common_files=''):
    if False:
        i = 10
        return i + 15
    return f'{introduction}\n\n{main_example}\n\n{app_prop}\n\n{auth}\n\n{auth_details}\n\n{props}\n\n{async_run}\n\n{http}\n\n{db}\n\n{hooks}\n\n{platform_axios}\n\n{async_options}\n\n{source_metadata}\n\n{common_files(parsed_common_files)}\n\n{typescript_definitions}\n\n{other_example}\n\n{rules}\n\n{additional_rules}\n\n{end}'