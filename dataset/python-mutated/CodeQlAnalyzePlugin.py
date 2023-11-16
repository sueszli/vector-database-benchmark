import json
import logging
import os
import yaml
from analyze import analyze_filter
from common import codeql_plugin
from edk2toolext import edk2_logging
from edk2toolext.environment.plugintypes.uefi_build_plugin import IUefiBuildPlugin
from edk2toolext.environment.uefi_build import UefiBuilder
from edk2toollib.uefi.edk2.path_utilities import Edk2Path
from edk2toollib.utility_functions import RunCmd
from pathlib import Path

class CodeQlAnalyzePlugin(IUefiBuildPlugin):

    def do_post_build(self, builder: UefiBuilder) -> int:
        if False:
            i = 10
            return i + 15
        'CodeQL analysis post-build functionality.\n\n        Args:\n            builder (UefiBuilder): A UEFI builder object for this build.\n\n        Returns:\n            int: The number of CodeQL errors found. Zero indicates that\n            AuditOnly mode is enabled or no failures were found.\n        '
        self.builder = builder
        self.package = builder.edk2path.GetContainingPackage(builder.edk2path.GetAbsolutePathOnThisSystemFromEdk2RelativePath(builder.env.GetValue('ACTIVE_PLATFORM')))
        self.package_path = Path(builder.edk2path.GetAbsolutePathOnThisSystemFromEdk2RelativePath(self.package))
        self.target = builder.env.GetValue('TARGET')
        self.codeql_db_path = codeql_plugin.get_codeql_db_path(builder.ws, self.package, self.target, new_path=False)
        self.codeql_path = codeql_plugin.get_codeql_cli_path()
        if not self.codeql_path:
            logging.critical('CodeQL build enabled but CodeQL CLI application not found.')
            return -1
        codeql_sarif_dir_path = self.codeql_db_path[:self.codeql_db_path.rindex('-')]
        codeql_sarif_dir_path = codeql_sarif_dir_path.replace('-db-', '-analysis-')
        self.codeql_sarif_path = os.path.join(codeql_sarif_dir_path, os.path.basename(self.codeql_db_path) + '.sarif')
        edk2_logging.log_progress(f'Analyzing {self.package} ({self.target}) CodeQL database at:\n           {self.codeql_db_path}')
        edk2_logging.log_progress(f'Results will be written to:\n           {self.codeql_sarif_path}')
        audit_only = False
        query_specifiers = None
        package_config_file = Path(os.path.join(self.package_path, self.package + '.ci.yaml'))
        plugin_data = None
        if package_config_file.is_file():
            with open(package_config_file, 'r') as cf:
                package_config_file_data = yaml.safe_load(cf)
                if 'CodeQlAnalyze' in package_config_file_data:
                    plugin_data = package_config_file_data['CodeQlAnalyze']
                    if 'AuditOnly' in plugin_data:
                        audit_only = plugin_data['AuditOnly']
                    if 'QuerySpecifiers' in plugin_data:
                        logging.debug(f'Loading CodeQL query specifiers in {str(package_config_file)}')
                        query_specifiers = plugin_data['QuerySpecifiers']
        global_audit_only = builder.env.GetValue('STUART_CODEQL_AUDIT_ONLY')
        if global_audit_only:
            if global_audit_only.strip().lower() == 'true':
                audit_only = True
        if audit_only:
            logging.info(f'CodeQL Analyze plugin is in audit only mode for {self.package} ({self.target}).')
        if not query_specifiers:
            query_specifiers = builder.env.GetValue('STUART_CODEQL_QUERY_SPECIFIERS')
        plugin_query_set = Path(Path(__file__).parent, 'CodeQlQueries.qls')
        if not query_specifiers and plugin_query_set.is_file():
            query_specifiers = str(plugin_query_set.resolve())
        if not query_specifiers:
            logging.warning('Skipping CodeQL analysis since no CodeQL query specifiers were provided.')
            return 0
        codeql_params = f'database analyze {self.codeql_db_path} {query_specifiers} --format=sarifv2.1.0 --output={self.codeql_sarif_path} --download --threads=0'
        Path(self.codeql_sarif_path).parent.mkdir(exist_ok=True, parents=True)
        cmd_ret = RunCmd(self.codeql_path, codeql_params)
        if cmd_ret != 0:
            logging.critical(f'CodeQL CLI analysis failed with return code {cmd_ret}.')
        if not os.path.isfile(self.codeql_sarif_path):
            logging.critical(f'The sarif file {self.codeql_sarif_path} was not created. Analysis cannot continue.')
            return -1
        filter_pattern_data = []
        global_filter_file_value = builder.env.GetValue('STUART_CODEQL_FILTER_FILES')
        if global_filter_file_value:
            global_filter_files = global_filter_file_value.strip().split(',')
            global_filter_files = [Path(f) for f in global_filter_files]
            for global_filter_file in global_filter_files:
                if global_filter_file.is_file():
                    with open(global_filter_file, 'r') as ff:
                        global_filter_file_data = yaml.safe_load(ff)
                        if 'Filters' in global_filter_file_data:
                            current_pattern_data = global_filter_file_data['Filters']
                            if type(current_pattern_data) is not list:
                                logging.critical(f'CodeQL pattern data must be a list of strings. Data in {str(global_filter_file.resolve())} is invalid. CodeQL analysis is incomplete.')
                                return -1
                            filter_pattern_data += current_pattern_data
                        else:
                            logging.critical(f'CodeQL global filter file {str(global_filter_file.resolve())} is  malformed. Missing Filters section. CodeQL analysis is incomplete.')
                            return -1
                else:
                    logging.critical(f'CodeQL global filter file {str(global_filter_file.resolve())} was not found. CodeQL analysis is incomplete.')
                    return -1
        if plugin_data and 'Filters' in plugin_data:
            if type(plugin_data['Filters']) is not list:
                logging.critical('CodeQL pattern data must be a list of strings. CodeQL analysis is incomplete.')
                return -1
            filter_pattern_data.extend(plugin_data['Filters'])
        if filter_pattern_data:
            logging.info('Applying CodeQL SARIF result filters.')
            analyze_filter.filter_sarif(self.codeql_sarif_path, self.codeql_sarif_path, filter_pattern_data, split_lines=False)
        with open(self.codeql_sarif_path, 'r') as sf:
            sarif_file_data = json.load(sf)
        try:
            total_errors = 0
            for run in sarif_file_data['runs']:
                total_errors += len(run['results'])
        except KeyError:
            logging.critical('Sarif file does not contain expected data. Analysis cannot continue.')
            return -1
        if total_errors > 0:
            if audit_only:
                logging.warning(f'{self.package} ({self.target}) CodeQL analysis ignored {total_errors} errors due to audit mode being enabled.')
                return 0
            else:
                logging.error(f'{self.package} ({self.target}) CodeQL analysis failed with {total_errors} errors.')
        return total_errors