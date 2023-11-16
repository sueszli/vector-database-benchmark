""" Isolates code download and merge logic for dynamic Schemas template """
import json
import os
import click
from botocore.exceptions import ClientError
from samcli.local.common.runtime_template import SAM_RUNTIME_TO_SCHEMAS_CODE_LANG_MAPPING
from samcli.local.lambdafn.zip import unzip

def do_download_source_code_binding(runtime, schema_template_details, schemas_api_caller, download_location):
    if False:
        while True:
            i = 10
    "\n    Downloads source code binding for given registry and schema version,\n    generating the code bindings if they haven't been generated first\n    :param runtime: Lambda runtime\n    :param schema_template_details: e.g: registry_name, schema_name, schema_version\n    :param schemas_api_caller: the schemas api caller object\n    :param download_location: the download location\n    :return: directory location where code is downloaded\n    "
    registry_name = schema_template_details['registry_name']
    schema_name = schema_template_details['schema_full_name']
    schema_version = schema_template_details['schema_version']
    schemas_runtime = SAM_RUNTIME_TO_SCHEMAS_CODE_LANG_MAPPING.get(runtime)
    try:
        click.echo('Event code binding Registry: %s and Schema: %s' % (registry_name, schema_name))
        click.echo('Generating code bindings...')
        return schemas_api_caller.download_source_code_binding(schemas_runtime, registry_name, schema_name, schema_version, download_location)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NotFoundException':
            schemas_api_caller.put_code_binding(schemas_runtime, registry_name, schema_name, schema_version)
            schemas_api_caller.poll_for_code_binding_status(schemas_runtime, registry_name, schema_name, schema_version)
            return schemas_api_caller.download_source_code_binding(schemas_runtime, registry_name, schema_name, schema_version, download_location)
        raise e

def do_extract_and_merge_schemas_code(download_location, output_dir, project_name, template_location):
    if False:
        print('Hello World!')
    '\n    Unzips schemas generated code and merge it with cookiecutter genertaed source.\n    :param download_location:\n    :param output_dir:\n    :param project_name:\n    :param template_location:\n    '
    click.echo('Merging code bindings...')
    cookiecutter_json_path = os.path.join(template_location, 'cookiecutter.json')
    with open(cookiecutter_json_path, 'r') as cookiecutter_file:
        cookiecutter_json_data = cookiecutter_file.read()
        cookiecutter_json = json.loads(cookiecutter_json_data)
        function_name = cookiecutter_json['function_name']
        copy_location = os.path.join(output_dir, project_name, function_name)
        unzip(download_location, copy_location)