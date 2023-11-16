import logging
import json
import yaml
from io import StringIO
import os
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toollib.utility_functions import RunCmd
from edk2toolext.environment.var_dict import VarDict
from edk2toollib.gitignore_parser import parse_gitignore_lines
from edk2toolext.environment import version_aggregator

class SpellCheck(ICiBuildPlugin):
    """
    A CiBuildPlugin that uses the cspell node module to scan the files
    from the package being tested for spelling errors.  The plugin contains
    the base cspell.json file then thru the configuration options other settings
    can be changed or extended.

    Configuration options:
    "SpellCheck": {
        "AuditOnly": False,          # Don't fail the build if there are errors.  Just log them
        "IgnoreFiles": [],           # use gitignore syntax to ignore errors in matching files
        "ExtendWords": [],           # words to extend to the dictionary for this package
        "IgnoreStandardPaths": [],   # Standard Plugin defined paths that should be ignore
        "AdditionalIncludePaths": [] # Additional paths to spell check (wildcards supported)
    }
    """
    STANDARD_PLUGIN_DEFINED_PATHS = ('*.c', '*.h', '*.nasm', '*.asm', '*.masm', '*.s', '*.asl', '*.dsc', '*.dec', '*.fdf', '*.inf', '*.md', '*.txt')

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            while True:
                i = 10
        ' Provide the testcase name and classname for use in reporting\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n                testclassname: a descriptive string for the testcase can include whitespace\n                classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n        '
        return ('Spell check files in ' + packagename, packagename + '.SpellCheck')

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            return 10
        Errors = []
        abs_pkg_path = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        if abs_pkg_path is None:
            tc.SetSkipped()
            tc.LogStdError('No package {0}'.format(packagename))
            return -1
        return_buffer = StringIO()
        ret = RunCmd('node', '--version', outstream=return_buffer)
        if ret != 0:
            tc.SetSkipped()
            tc.LogStdError("NodeJs not installed. Test can't run")
            logging.warning("NodeJs not installed. Test can't run")
            return -1
        node_version = return_buffer.getvalue().strip()
        tc.LogStdOut(f'Node version: {node_version}')
        version_aggregator.GetVersionAggregator().ReportVersion('NodeJs', node_version, version_aggregator.VersionTypes.INFO)
        return_buffer = StringIO()
        ret = RunCmd('cspell', '--version', outstream=return_buffer)
        if ret != 0:
            tc.SetSkipped()
            tc.LogStdError("cspell not installed.  Test can't run")
            logging.warning("cspell not installed.  Test can't run")
            return -1
        cspell_version = return_buffer.getvalue().strip()
        tc.LogStdOut(f'CSpell version: {cspell_version}')
        version_aggregator.GetVersionAggregator().ReportVersion('CSpell', cspell_version, version_aggregator.VersionTypes.INFO)
        package_relative_paths_to_spell_check = list(SpellCheck.STANDARD_PLUGIN_DEFINED_PATHS)
        if 'IgnoreStandardPaths' in pkgconfig:
            for a in pkgconfig['IgnoreStandardPaths']:
                if a in package_relative_paths_to_spell_check:
                    tc.LogStdOut(f'ignoring standard path due to ci.yaml ignore: {a}')
                    package_relative_paths_to_spell_check.remove(a)
                else:
                    tc.LogStdOut(f'Invalid IgnoreStandardPaths value: {a}')
        if 'AdditionalIncludePaths' in pkgconfig:
            package_relative_paths_to_spell_check.extend(pkgconfig['AdditionalIncludePaths'])
        relpath = os.path.relpath(abs_pkg_path)
        cpsell_paths = ' '.join([f'"{relpath}/**/{x}"' for x in package_relative_paths_to_spell_check])
        config_file_path = os.path.join(Edk2pathObj.WorkspacePath, 'Build', packagename, 'cspell_actual_config.json')
        mydir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(mydir, 'cspell.base.yaml')
        with open(base, 'r') as i:
            config = yaml.safe_load(i)
        if 'ExtendWords' in pkgconfig:
            config['words'].extend(pkgconfig['ExtendWords'])
        with open(config_file_path, 'w') as o:
            json.dump(config, o)
        All_Ignores = []
        if 'IgnoreFiles' in pkgconfig:
            All_Ignores.extend(pkgconfig['IgnoreFiles'])
        ignore = parse_gitignore_lines(All_Ignores, os.path.join(abs_pkg_path, 'nofile.txt'), abs_pkg_path)
        EasyFix = []
        results = self._check_spelling(cpsell_paths, config_file_path)
        for r in results:
            (path, _, word) = r.partition(' - Unknown word ')
            if len(word) == 0:
                continue
            pathinfo = path.rsplit(':', 2)
            if ignore(pathinfo[0]):
                tc.LogStdOut(f'ignoring error due to ci.yaml ignore: {r}')
                continue
            EasyFix.append(word.strip().strip('()'))
            Errors.append(r)
        for l in Errors:
            tc.LogStdError(l.strip())
        if len(EasyFix) > 0:
            EasyFix = sorted(set((a.lower() for a in EasyFix)))
            tc.LogStdOut('\n Easy fix:')
            OneString = 'If these are not errors add this to your ci.yaml file.\n'
            OneString += '"SpellCheck": {\n  "ExtendWords": ['
            for a in EasyFix:
                tc.LogStdOut(f'\n"{a}",')
                OneString += f'\n    "{a}",'
            logging.info(OneString.rstrip(',') + '\n  ]\n}')
        overall_status = len(Errors)
        if overall_status != 0:
            if 'AuditOnly' in pkgconfig and pkgconfig['AuditOnly']:
                tc.SetSkipped()
                return -1
            else:
                tc.SetFailed('SpellCheck {0} Failed.  Errors {1}'.format(packagename, overall_status), 'CHECK_FAILED')
        else:
            tc.SetSuccess()
        return overall_status

    def _check_spelling(self, abs_file_to_check: str, abs_config_file_to_use: str) -> []:
        if False:
            for i in range(10):
                print('nop')
        output = StringIO()
        ret = RunCmd('cspell', f'--config {abs_config_file_to_use} {abs_file_to_check}', outstream=output)
        if ret == 0:
            return []
        else:
            return output.getvalue().strip().splitlines()