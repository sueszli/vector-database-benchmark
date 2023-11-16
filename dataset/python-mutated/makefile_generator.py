"""
Terraform Makefile and make rule generation

This module generates the Makefile for the project and the rules for each of the Lambda functions found
"""
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from samcli.hook_packages.terraform.hooks.prepare.types import SamMetadataResource
from samcli.lib.utils.path_utils import convert_path_to_unix_path
LOG = logging.getLogger(__name__)
TERRAFORM_BUILD_SCRIPT = 'copy_terraform_built_artifacts.py'
ZIP_UTILS_MODULE = 'zip.py'
TF_BACKEND_OVERRIDE_FILENAME = 'z_samcli_backend_override'

def generate_makefile_rule_for_lambda_resource(sam_metadata_resource: SamMetadataResource, logical_id: str, terraform_application_dir: str, python_command_name: str, output_dir: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Generates and returns a makefile rule for the lambda resource associated with the given sam metadata resource.\n\n    Parameters\n    ----------\n    sam_metadata_resource: SamMetadataResource\n        A sam metadata resource; the generated makefile rule will correspond to building the lambda resource\n        associated with this sam metadata resource\n    logical_id: str\n        Logical ID of the lambda resource\n    terraform_application_dir: str\n        the terraform project root directory\n    python_command_name: str\n        the python command name to use for running a script in the makefile rule\n    output_dir: str\n        the directory into which the Makefile is written\n\n    Returns\n    -------\n    str\n        The generated makefile rule\n    '
    target = _get_makefile_build_target(logical_id)
    resource_address = sam_metadata_resource.resource.get('address', '')
    python_command_recipe = _format_makefile_recipe(_build_makerule_python_command(python_command_name, output_dir, resource_address, sam_metadata_resource, terraform_application_dir))
    return f'{target}{python_command_recipe}'

def generate_makefile(makefile_rules: List[str], output_directory_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates a makefile with the given rules in the given directory\n\n    Parameters\n    ----------\n    makefile_rules: List[str]\n        the list of rules to write in the Makefile\n    output_directory_path: str\n        the output directory path to write the generated makefile\n    '
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path, exist_ok=True)
    _generate_backend_override_file(output_directory_path)
    copy_terraform_built_artifacts_script_path = os.path.join(Path(os.path.dirname(__file__)).parent.parent, TERRAFORM_BUILD_SCRIPT)
    shutil.copy(copy_terraform_built_artifacts_script_path, output_directory_path)
    samcli_root_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent
    ZIP_UTILS_MODULE_script_path = os.path.join(samcli_root_path, 'local', 'lambdafn', ZIP_UTILS_MODULE)
    shutil.copy(ZIP_UTILS_MODULE_script_path, output_directory_path)
    makefile_path = os.path.join(output_directory_path, 'Makefile')
    with open(makefile_path, 'w+') as makefile:
        makefile.writelines(makefile_rules)

def _generate_backend_override_file(output_directory_path: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates an override tf file to use a temporary backend\n\n    Parameters\n    ----------\n    output_directory_path: str\n        the output directory path to write the generated makefile\n    '
    statefile_filename = f'{uuid.uuid4()}.tfstate'
    override_content = f'terraform {{\n  backend "local" {{\n    path = "./{statefile_filename}"\n  }}\n}}\n'
    override_file_path = os.path.join(output_directory_path, TF_BACKEND_OVERRIDE_FILENAME)
    with open(override_file_path, 'w+') as f:
        f.write(override_content)

def _build_makerule_python_command(python_command_name: str, output_dir: str, resource_address: str, sam_metadata_resource: SamMetadataResource, terraform_application_dir: str) -> str:
    if False:
        return 10
    '\n    Build the Python command recipe to be used inside of the Makefile rule\n\n    Parameters\n    ----------\n    python_command_name: str\n        the python command name to use for running a script in the makefile recipe\n    output_dir: str\n        the directory into which the Makefile is written\n    resource_address: str\n        Address of a given terraform resource\n    sam_metadata_resource: SamMetadataResource\n        A sam metadata resource; the generated show command recipe will correspond to building the lambda resource\n        associated with this sam metadata resource\n    terraform_application_dir: str\n        the terraform project root directory\n\n    Returns\n    -------\n    str\n        Fully resolved Terraform show command\n    '
    show_command_template = '{python_command_name} "{terraform_built_artifacts_script_path}" --expression "{jpath_string}" --directory "$(ARTIFACTS_DIR)" --target "{resource_address}"'
    jpath_string = _build_jpath_string(sam_metadata_resource, resource_address)
    terraform_built_artifacts_script_path = convert_path_to_unix_path(str(Path(output_dir, TERRAFORM_BUILD_SCRIPT).relative_to(terraform_application_dir)))
    return show_command_template.format(python_command_name=python_command_name, terraform_built_artifacts_script_path=terraform_built_artifacts_script_path, jpath_string=jpath_string.replace('"', '\\"'), resource_address=resource_address.replace('"', '\\"'))

def _get_makefile_build_target(logical_id: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Formats the Makefile rule build target string as is needed by the Makefile\n\n    Parameters\n    ----------\n    logical_id: str\n       Logical ID of the resource to use for the Makefile rule target\n\n    Returns\n    -------\n    str\n        The formatted Makefile rule build target\n    '
    return f'build-{logical_id}:\n'

def _format_makefile_recipe(rule_string: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Formats the Makefile rule string as is needed by the Makefile\n\n    Parameters\n    ----------\n    rule_string: str\n       Makefile rule string to be formatted\n\n    Returns\n    -------\n    str\n        The formatted target rule\n    '
    return f'\t{rule_string}\n'

def _build_jpath_string(sam_metadata_resource: SamMetadataResource, resource_address: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Constructs the JPath string for a given sam metadata resource from the planned_values\n    to the build_output_path as is created by the Terraform plan output\n\n    Parameters\n    ----------\n    sam_metadata_resource: SamMetadataResource\n        A sam metadata resource; the generated recipe jpath will correspond to building the lambda resource\n        associated with this sam metadata resource\n\n    resource_address: str\n        Full address of a Terraform resource\n\n    Returns\n    -------\n    str\n       Full JPath string for a resource from planned_values to build_output_path\n    '
    jpath_string_template = '|values|root_module{child_modules}|resources|[?address=="{resource_address}"]|values|triggers|built_output_path'
    child_modules_template = '|child_modules|[?address=={module_address}]'
    module_address = sam_metadata_resource.current_module_address
    full_module_path = ''
    parent_modules = _get_parent_modules(module_address)
    for module in parent_modules:
        full_module_path += child_modules_template.format(module_address=module)
    jpath_string = jpath_string_template.format(child_modules=full_module_path, resource_address=resource_address)
    return jpath_string

def _get_parent_modules(module_address: Optional[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an a full Terraform resource address to a list of module\n    addresses from the root module to the current module\n\n    e.g. "module.level1_lambda.module.level2_lambda" as input will return\n    ["module.level1_lambda", "module.level1_lambda.module.level2_lambda"]\n\n    Parameters\n    ----------\n    module_address: str\n       Full address of the Terraform module\n\n    Returns\n    -------\n    List[str]\n       List of module addresses starting from the root module to the current module\n    '
    if not module_address:
        return []
    modules = module_address.split('.')
    modules = ['.'.join(modules[i:i + 2]) for i in range(0, len(modules), 2)]
    if not modules:
        return []
    previous_module = modules[0]
    full_path_modules = [previous_module]
    for module in modules[1:]:
        norm_module = previous_module + '.' + module
        previous_module = norm_module
        full_path_modules.append(norm_module)
    return full_path_modules