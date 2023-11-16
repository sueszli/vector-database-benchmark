from templates.actions.additional_rules import additional_rules
from templates.actions.export_summary import export_summary
from templates.actions.introduction import introduction
from templates.actions.main_example import main_example
from templates.actions.other_example import other_example
from templates.common.app_prop import app_prop
from templates.common.auth import auth
from templates.common.common_files import common_files
from templates.common.component_metadata import action_metadata
from templates.common.platform_axios import platform_axios
from templates.common.props import props
from templates.common.rules import rules
from templates.common.async_options import async_options
from templates.common.typescript_definitions import typescript_definitions
from templates.common.end import end
checks = [app_prop, auth, props, export_summary, platform_axios, async_options, action_metadata, rules, additional_rules, typescript_definitions]
always_include = [introduction, typescript_definitions, main_example, other_example, end]

def system_instructions(auth_details='', parsed_common_files=''):
    if False:
        return 10
    return f'{introduction}\n\n{main_example}\n\n{other_example}\n\n{app_prop}\n\n{auth}\n\n{auth_details}\n\n{props}\n\n{export_summary}\n\n{platform_axios}\n\n{async_options}\n\n{action_metadata}\n\n{common_files(parsed_common_files)}\n\n{typescript_definitions}\n\n{rules}\n\n{additional_rules}\n\n{end}'