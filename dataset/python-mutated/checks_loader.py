from colorama import Fore, Style
from prowler.lib.check.check import parse_checks_from_compliance_framework, parse_checks_from_file, recover_checks_from_provider, recover_checks_from_service
from prowler.lib.logger import logger

def load_checks_to_execute(bulk_checks_metadata: dict, bulk_compliance_frameworks: dict, checks_file: str, check_list: list, service_list: list, severities: list, compliance_frameworks: list, categories: set, provider: str) -> set:
    if False:
        return 10
    'Generate the list of checks to execute based on the cloud provider and input arguments specified'
    checks_to_execute = set()
    if check_list:
        for check_name in check_list:
            checks_to_execute.add(check_name)
    elif severities:
        for check in bulk_checks_metadata:
            if bulk_checks_metadata[check].Severity in severities:
                checks_to_execute.add(check)
        if service_list:
            checks_to_execute = recover_checks_from_service(service_list, provider) & checks_to_execute
    elif checks_file:
        try:
            checks_to_execute = parse_checks_from_file(checks_file, provider)
        except Exception as e:
            logger.error(f'{e.__class__.__name__}[{e.__traceback__.tb_lineno}] -- {e}')
    elif service_list:
        checks_to_execute = recover_checks_from_service(service_list, provider)
    elif compliance_frameworks:
        try:
            checks_to_execute = parse_checks_from_compliance_framework(compliance_frameworks, bulk_compliance_frameworks)
        except Exception as e:
            logger.error(f'{e.__class__.__name__}[{e.__traceback__.tb_lineno}] -- {e}')
    elif categories:
        for cat in categories:
            for check in bulk_checks_metadata:
                if cat in bulk_checks_metadata[check].Categories:
                    checks_to_execute.add(check)
    else:
        try:
            checks = recover_checks_from_provider(provider)
        except Exception as e:
            logger.error(f'{e.__class__.__name__}[{e.__traceback__.tb_lineno}] -- {e}')
        else:
            for check_info in checks:
                check_name = check_info[0]
                checks_to_execute.add(check_name)
    check_aliases = {}
    for (check, metadata) in bulk_checks_metadata.items():
        for alias in metadata.CheckAliases:
            check_aliases[alias] = check
    for input_check in checks_to_execute:
        if input_check in check_aliases and check_aliases[input_check] not in checks_to_execute:
            checks_to_execute.remove(input_check)
            checks_to_execute.add(check_aliases[input_check])
            print(f'\nUsing alias {Fore.YELLOW}{input_check}{Style.RESET_ALL} for check {Fore.YELLOW}{check_aliases[input_check]}{Style.RESET_ALL}...\n')
    return checks_to_execute