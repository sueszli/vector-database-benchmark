import os
import re
import shlex
from functools import reduce
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import simplejson
from jinja2 import Template
from mage_ai.data_preparation.models.block.dbt import DBTBlock
from mage_ai.data_preparation.models.block.dbt.dbt_cli import DBTCli
from mage_ai.data_preparation.models.block.dbt.profiles import Profiles
from mage_ai.data_preparation.models.block.dbt.project import Project
from mage_ai.data_preparation.shared.utils import get_template_vars
from mage_ai.data_preparation.templates.utils import get_variable_for_template
from mage_ai.orchestration.constants import PIPELINE_RUN_MAGE_VARIABLES_KEY
from mage_ai.shared.hash import merge_dict

class DBTBlockYAML(DBTBlock):

    @property
    def project_path(self) -> Union[str, os.PathLike]:
        if False:
            while True:
                i = 10
        '\n        Gets the path of the dbt project in use.\n\n        Returns:\n            Union[str, os.PathLike]: Path of the dbt project, being used\n        '
        project_name = self.configuration.get('dbt_project_name')
        if project_name:
            return str(Path(self.base_project_path) / project_name)

    def tags(self) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Get the tags associated with the DBT block.\n\n        Returns:\n            List[str]: The list of tags.\n        '
        arr = super().tags()
        command = self._dbt_configuration.get('command', 'run')
        if command:
            arr.append(command)
        return arr

    async def metadata_async(self) -> Dict[str, Any]:
        """
        Retrieves metadata needed to configure the block.
        - available local dbt projects
        - target to use and other target of the local dbt projects

        Returns:
            Dict: The metadata of the DBT block.
        """
        projects = {}
        project_dirs = Project(self.base_project_path).local_packages
        for project_dir in project_dirs:
            project_full_path = str(Path(self.base_project_path) / project_dir)
            project = Project(project_full_path).project
            project_name = project.get('name')
            project_profile = project.get('profile')
            try:
                profiles = Profiles(project_full_path, self.pipeline.variables if self.pipeline else None).profiles
            except FileNotFoundError:
                profiles = None
            profile = profiles.get(project_profile) if profiles else None
            target = profile.get('target') if profile else None
            targets = sorted(list(profile.get('outputs').keys())) if profile else None
            projects[project_dir] = {'project_name': project_name, 'target': target, 'targets': targets}
        return {'dbt': {'block': {}, 'project': None, 'projects': projects}}

    def _hydrate_block_outputs(self, content: str, outputs_from_input_vars: Dict, variables: Dict=None) -> str:
        if False:
            return 10

        def _block_output(block_uuid: str=None, parse: str=None, outputs_from_input_vars=outputs_from_input_vars, upstream_block_uuids=self.upstream_block_uuids, variables=variables) -> Any:
            if False:
                print('Hello World!')
            data = outputs_from_input_vars
            if parse:
                if block_uuid:
                    data = outputs_from_input_vars.get(block_uuid)
                elif upstream_block_uuids:

                    def _build_positional_arguments(acc: List, upstream_block_uuid: str) -> List:
                        if False:
                            i = 10
                            return i + 15
                        acc.append(outputs_from_input_vars.get(upstream_block_uuid))
                        return acc
                    data = reduce(_build_positional_arguments, upstream_block_uuids, [])
                try:
                    return parse(data, variables)
                except Exception as err:
                    print(f'[WARNING] block_output: {err}')
            return data
        variable_pattern = '{}[ ]*block_output[^{}]*[ ]*{}'.format('\\{\\{', '\\}\\}', '\\}\\}')
        match = 1
        while match is not None:
            match = None
            match = re.search(variable_pattern, content, re.IGNORECASE)
            if not match:
                continue
            (si, ei) = match.span()
            substring = content[si:ei]
            match2 = re.match('{}([^{}{}]+){}'.format('\\{\\{', '\\{\\{', '\\}\\}', '\\}\\}'), substring)
            if match2:
                groups = match2.groups()
                if groups:
                    function_string = groups[0].strip()
                    results = dict(block_output=_block_output)
                    exec(f'value_hydrated = {function_string}', results)
                    value_hydrated = results['value_hydrated']
                    content = f'{content[0:si]}{value_hydrated}{content[ei:]}'
        return content

    def _execute_block(self, outputs_from_input_vars, global_vars: Optional[Dict[str, Any]]=None, logger: Logger=None, runtime_arguments: Optional[Dict[str, Any]]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Execute the DBT block.\n\n        Args:\n            outputs_from_input_vars:\n            global_vars (Optional[Dict[str, Any]], optional): The global variables.\n            runtime_arguments (Optional[Dict[str, Any]], optional): The runtime arguments.\n        '
        task = self._dbt_configuration.get('command', 'run')
        variables = merge_dict(global_vars, runtime_arguments or {})
        content = self.content or ''
        content = self._hydrate_block_outputs(content, outputs_from_input_vars, variables)
        content = Template(content).render(variables=lambda x, p=None, v=variables: get_variable_for_template(x, parse=p, variables=v), **get_template_vars())
        args = shlex.split(content)
        args += ['--project-dir', self.project_path]
        runtime_configuration = variables.get(PIPELINE_RUN_MAGE_VARIABLES_KEY, {}).get('blocks', {}).get(self.uuid, {}).get('configuration', {})
        if runtime_configuration.get('flags'):
            flags = runtime_configuration['flags']
            args += flags if isinstance(flags, list) else [flags]
        vars_index = None
        try:
            vars_index = args.index('--vars') + 1
        except Exception:
            pass
        if vars_index:
            variables2 = Template(args[vars_index]).render(variables=lambda x, p=None, v=variables: get_variable_for_template(x, parse=p, variables=v), **get_template_vars())
            args[vars_index] = self._variables_json(merge_dict(variables, simplejson.loads(variables2)))
        else:
            args += ['--vars', self._variables_json(variables)]
        target = self.configuration.get('dbt_profile_target')
        if target:
            target = Template(target).render(variables=lambda x: variables.get(x) if variables else None, **get_template_vars())
            args += ['--target', target]
        with Profiles(self.project_path, variables) as profiles:
            args += ['--profiles-dir', str(profiles.profiles_dir)]
            (_res, success) = DBTCli([task] + args, logger).invoke()
            if not success:
                raise Exception('DBT exited with a non 0 exit status.')