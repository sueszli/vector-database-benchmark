"""Providers sub-commands."""
from __future__ import annotations
import sys
import re2
from airflow.cli.simple_table import AirflowConsole
from airflow.providers_manager import ProvidersManager
from airflow.utils.cli import suppress_logs_and_warning
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
ERROR_IMPORTING_HOOK = 'Error when importing hook!'

def _remove_rst_syntax(value: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return re2.sub('[`_<>]', '', value.strip(' \n.'))

@suppress_logs_and_warning
@providers_configuration_loaded
def provider_get(args):
    if False:
        return 10
    'Get a provider info.'
    providers = ProvidersManager().providers
    if args.provider_name in providers:
        provider_version = providers[args.provider_name].version
        provider_info = providers[args.provider_name].data
        if args.full:
            provider_info['description'] = _remove_rst_syntax(provider_info['description'])
            AirflowConsole().print_as(data=[provider_info], output=args.output)
        else:
            AirflowConsole().print_as(data=[{'Provider': args.provider_name, 'Version': provider_version}], output=args.output)
    else:
        raise SystemExit(f'No such provider installed: {args.provider_name}')

@suppress_logs_and_warning
@providers_configuration_loaded
def providers_list(args):
    if False:
        i = 10
        return i + 15
    'List all providers at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().providers.values()), output=args.output, mapper=lambda x: {'package_name': x.data['package-name'], 'description': _remove_rst_syntax(x.data['description']), 'version': x.version})

@suppress_logs_and_warning
@providers_configuration_loaded
def hooks_list(args):
    if False:
        i = 10
        return i + 15
    'List all hooks at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().hooks.items()), output=args.output, mapper=lambda x: {'connection_type': x[0], 'class': x[1].hook_class_name if x[1] else ERROR_IMPORTING_HOOK, 'conn_id_attribute_name': x[1].connection_id_attribute_name if x[1] else ERROR_IMPORTING_HOOK, 'package_name': x[1].package_name if x[1] else ERROR_IMPORTING_HOOK, 'hook_name': x[1].hook_name if x[1] else ERROR_IMPORTING_HOOK})

@suppress_logs_and_warning
@providers_configuration_loaded
def triggers_list(args):
    if False:
        print('Hello World!')
    AirflowConsole().print_as(data=ProvidersManager().trigger, output=args.output, mapper=lambda x: {'package_name': x.package_name, 'class': x.trigger_class_name, 'integration_name': x.integration_name})

@suppress_logs_and_warning
@providers_configuration_loaded
def notifications_list(args):
    if False:
        i = 10
        return i + 15
    AirflowConsole().print_as(data=ProvidersManager().notification, output=args.output, mapper=lambda x: {'notification_class_name': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def connection_form_widget_list(args):
    if False:
        i = 10
        return i + 15
    'List all custom connection form fields at the command line.'
    AirflowConsole().print_as(data=sorted(ProvidersManager().connection_form_widgets.items()), output=args.output, mapper=lambda x: {'connection_parameter_name': x[0], 'class': x[1].hook_class_name, 'package_name': x[1].package_name, 'field_type': x[1].field.field_class.__name__})

@suppress_logs_and_warning
@providers_configuration_loaded
def connection_field_behaviours(args):
    if False:
        print('Hello World!')
    'List field behaviours.'
    AirflowConsole().print_as(data=list(ProvidersManager().field_behaviours), output=args.output, mapper=lambda x: {'field_behaviours': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def extra_links_list(args):
    if False:
        while True:
            i = 10
    'List all extra links at the command line.'
    AirflowConsole().print_as(data=ProvidersManager().extra_links_class_names, output=args.output, mapper=lambda x: {'extra_link_class_name': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def logging_list(args):
    if False:
        print('Hello World!')
    'List all log task handlers at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().logging_class_names), output=args.output, mapper=lambda x: {'logging_class_name': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def secrets_backends_list(args):
    if False:
        return 10
    'List all secrets backends at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().secrets_backend_class_names), output=args.output, mapper=lambda x: {'secrets_backend_class_name': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def auth_backend_list(args):
    if False:
        i = 10
        return i + 15
    'List all API auth backend modules at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().auth_backend_module_names), output=args.output, mapper=lambda x: {'api_auth_backand_module': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def executors_list(args):
    if False:
        return 10
    'List all executors at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().executor_class_names), output=args.output, mapper=lambda x: {'executor_class_names': x})

@suppress_logs_and_warning
@providers_configuration_loaded
def config_list(args):
    if False:
        while True:
            i = 10
    'List all configurations at the command line.'
    AirflowConsole().print_as(data=list(ProvidersManager().provider_configs), output=args.output, mapper=lambda x: {'provider_config': x})

@suppress_logs_and_warning
def lazy_loaded(args):
    if False:
        print('Hello World!')
    'Informs if providers manager has been initialized too early.\n\n    If provider is initialized, shows the stack trace and exit with error code 1.\n    '
    import rich
    if ProvidersManager.initialized():
        rich.print('\n[red]ProvidersManager was initialized during CLI parsing. This should not happen.\n', file=sys.stderr)
        rich.print('\n[yellow]Please make sure no Providers Manager initialization happens during CLI parsing.\n', file=sys.stderr)
        rich.print('Stack trace where it has been initialized:\n', file=sys.stderr)
        rich.print(ProvidersManager.initialization_stack_trace(), file=sys.stderr)
        sys.exit(1)
    else:
        rich.print('[green]All ok. Providers Manager was not initialized during the CLI parsing.')
        sys.exit(0)