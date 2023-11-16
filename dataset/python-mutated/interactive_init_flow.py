"""
Isolates interactive init prompt flow. Expected to call generator logic at end of flow.
"""
import logging
import pathlib
import tempfile
from typing import Optional, Tuple
import click
from botocore.exceptions import ClientError, WaiterError
from samcli.commands._utils.options import generate_next_command_recommendation
from samcli.commands.exceptions import InvalidInitOptionException, SchemasApiException
from samcli.commands.init.init_flow_helpers import _get_image_from_runtime, _get_runtime_from_image, _get_templates_with_dependency_manager, get_architectures, get_sorted_runtimes
from samcli.commands.init.init_generator import do_generate
from samcli.commands.init.init_templates import InitTemplates, InvalidInitTemplateError
from samcli.commands.init.interactive_event_bridge_flow import get_schema_template_details, get_schemas_api_caller, get_schemas_template_parameter
from samcli.lib.config.samconfig import DEFAULT_CONFIG_FILE_NAME
from samcli.lib.schemas.schemas_code_manager import do_download_source_code_binding, do_extract_and_merge_schemas_code
from samcli.lib.utils.osutils import remove
from samcli.lib.utils.packagetype import IMAGE, ZIP
from samcli.local.common.runtime_template import LAMBDA_IMAGES_RUNTIMES_MAP, get_provided_runtime_from_custom_runtime, is_custom_runtime
LOG = logging.getLogger(__name__)

def do_interactive(location, pt_explicit, package_type, runtime, architecture, base_image, dependency_manager, output_dir, name, app_template, no_input, tracing, application_insights):
    if False:
        i = 10
        return i + 15
    '\n    Implementation of the ``cli`` method when --interactive is provided.\n    It will ask customers a few questions to init a template.\n    '
    if app_template:
        location_opt_choice = '1'
    else:
        click.echo('Which template source would you like to use?')
        click.echo('\t1 - AWS Quick Start Templates\n\t2 - Custom Template Location')
        location_opt_choice = click.prompt('Choice', type=click.Choice(['1', '2']), show_choices=False)
    generate_application(location, pt_explicit, package_type, runtime, architecture, base_image, dependency_manager, output_dir, name, app_template, no_input, location_opt_choice, tracing, application_insights)

def generate_application(location, pt_explicit, package_type, runtime, architecture, base_image, dependency_manager, output_dir, name, app_template, no_input, location_opt_choice, tracing, application_insights):
    if False:
        return 10
    "\n    The method holds the decision logic for generating an application\n    Parameters\n    ----------\n    location : str\n        Location to SAM template\n    pt_explicit : bool\n        boolean representing if the customer explicitly stated packageType\n    package_type : str\n        Zip or Image\n    runtime : str\n        AWS Lambda runtime or Custom runtime\n    architecture : str\n        The architecture type 'x86_64' and 'arm64' in AWS\n    base_image : str\n        AWS Lambda base image\n    dependency_manager : str\n        Runtime's Dependency manager\n    output_dir : str\n        Project output directory\n    name : str\n        name of the project\n    app_template : str\n        AWS Serverless Application template\n    no_input : bool\n        Whether to prompt for input or to accept default values\n        (the default is False, which prompts the user for values it doesn't know for baking)\n    location_opt_choice : int\n        User input for selecting how to get customer a vended serverless application\n    tracing : bool\n        boolen value to determine if X-Ray tracing show be activated or not\n    application_insights : bool\n        boolean value to determine if AppInsights monitoring should be enabled or not\n    "
    if location_opt_choice == '1':
        _generate_from_use_case(location, pt_explicit, package_type, runtime, base_image, dependency_manager, output_dir, name, app_template, architecture, tracing, application_insights)
    else:
        _generate_from_location(location, package_type, runtime, dependency_manager, output_dir, name, no_input, tracing, application_insights)

def _generate_from_location(location, package_type, runtime, dependency_manager, output_dir, name, no_input, tracing, application_insights):
    if False:
        return 10
    location = click.prompt('\nTemplate location (git, mercurial, http(s), zip, path)', type=str)
    summary_msg = '\n-----------------------\nGenerating application:\n-----------------------\nLocation: {location}\nOutput Directory: {output_dir}\n    '.format(location=location, output_dir=output_dir)
    click.echo(summary_msg)
    do_generate(location, package_type, runtime, dependency_manager, output_dir, name, no_input, None, tracing, application_insights)

def _generate_from_use_case(location: Optional[str], pt_explicit: bool, package_type: Optional[str], runtime: Optional[str], base_image: Optional[str], dependency_manager: Optional[str], output_dir: Optional[str], name: Optional[str], app_template: Optional[str], architecture: Optional[str], tracing: Optional[bool], application_insights: Optional[bool]) -> None:
    if False:
        i = 10
        return i + 15
    templates = InitTemplates()
    runtime_or_base_image = runtime if runtime else base_image
    package_type_filter_value = package_type if pt_explicit else None
    preprocessed_options = templates.get_preprocessed_manifest(runtime_or_base_image, app_template, package_type_filter_value, dependency_manager)
    question = 'Choose an AWS Quick Start application template'
    use_case = _get_choice_from_options(None, preprocessed_options, question, 'Template')
    default_app_template_properties = _generate_default_hello_world_application(use_case, package_type, runtime, base_image, dependency_manager, pt_explicit)
    chosen_app_template_properties = _get_app_template_properties(preprocessed_options, use_case, base_image, default_app_template_properties)
    (runtime, base_image, package_type, dependency_manager, template_chosen) = chosen_app_template_properties
    if tracing is None:
        tracing = prompt_user_to_enable_tracing()
    if application_insights is None:
        application_insights = prompt_user_to_enable_application_insights()
    app_template = template_chosen['appTemplate']
    base_image = LAMBDA_IMAGES_RUNTIMES_MAP.get(str(runtime)) if not base_image and package_type == IMAGE else base_image
    if not name:
        name = click.prompt('\nProject name', type=str, default='sam-app')
    location = templates.location_from_app_template(package_type, runtime, base_image, dependency_manager, app_template)
    final_architecture = get_architectures(architecture)
    lambda_supported_runtime = get_provided_runtime_from_custom_runtime(runtime) if is_custom_runtime(runtime) else runtime
    extra_context = {'project_name': name, 'runtime': lambda_supported_runtime, 'architectures': {'value': final_architecture}}
    is_dynamic_schemas_template = templates.is_dynamic_schemas_template(package_type, app_template, runtime, base_image, dependency_manager)
    if is_dynamic_schemas_template:
        schemas_api_caller = get_schemas_api_caller()
        schema_template_details = _get_schema_template_details(schemas_api_caller)
        schemas_template_parameter = get_schemas_template_parameter(schema_template_details)
        extra_context = {**schemas_template_parameter, **extra_context}
    no_input = True
    summary_msg = generate_summary_message(package_type, runtime, base_image, dependency_manager, output_dir, name, app_template, final_architecture)
    click.echo(summary_msg)
    command_suggestions = generate_next_command_recommendation([('Create pipeline', f'cd {name} && sam pipeline init --bootstrap'), ('Validate SAM template', f'cd {name} && sam validate'), ('Test Function in the Cloud', f'cd {name} && sam sync --stack-name {{stack-name}} --watch')])
    click.secho(command_suggestions, fg='yellow')
    do_generate(location, package_type, lambda_supported_runtime, dependency_manager, output_dir, name, no_input, extra_context, tracing, application_insights)
    if is_dynamic_schemas_template:
        _package_schemas_code(lambda_supported_runtime, schemas_api_caller, schema_template_details, output_dir, name, location)

def _generate_default_hello_world_application(use_case: str, package_type: Optional[str], runtime: Optional[str], base_image: Optional[str], dependency_manager: Optional[str], pt_explicit: bool) -> Tuple:
    if False:
        for i in range(10):
            print('nop')
    "\n    Generate the default Hello World template if Hello World Example is selected\n\n    Parameters\n    ----------\n    use_case : str\n        Type of template example selected\n    package_type : Optional[str]\n        The package type, 'Zip' or 'Image', see samcli/lib/utils/packagetype.py\n    runtime : Optional[str]\n        AWS Lambda function runtime\n    base_image : Optional[str]\n        AWS Lambda function base-image\n    dependency_manager : Optional[str]\n        dependency manager\n    pt_explicit : bool\n        True --package-type was passed or Vice versa\n\n    Returns\n    -------\n    Tuple\n        configuration for a default Hello World Example\n    "
    is_package_type_image = bool(package_type == IMAGE)
    if use_case == 'Hello World Example' and (not (runtime or base_image or is_package_type_image or dependency_manager)):
        if click.confirm('\nUse the most popular runtime and package type? (Python and zip)'):
            (runtime, package_type, dependency_manager, pt_explicit) = ('python3.9', ZIP, 'pip', True)
    return (runtime, package_type, dependency_manager, pt_explicit)

def _get_app_template_properties(preprocessed_options: dict, use_case: str, base_image: Optional[str], template_properties: Tuple) -> Tuple:
    if False:
        i = 10
        return i + 15
    '\n    This is the heart of the interactive flow, this method fetchs the templates options needed to generate a template\n\n    Parameters\n    ----------\n    preprocessed_options : dict\n        Preprocessed manifest from https://github.com/aws/aws-sam-cli-app-templates\n    use_case : Optional[str]\n        Type of template example selected\n    base_image : str\n        AWS Lambda function base-image\n    template_properties : Tuple\n        Tuple of template properties like runtime, packages type and dependency manager\n\n    Returns\n    -------\n    Tuple\n        Tuple of template configuration and the chosen template\n\n    Raises\n    ------\n    InvalidInitOptionException\n        exception raised when invalid option is provided\n    '
    (runtime, package_type, dependency_manager, pt_explicit) = template_properties
    runtime_options = preprocessed_options[use_case]
    runtime = None if is_custom_runtime(runtime) else runtime
    if not runtime and (not base_image):
        question = 'Which runtime would you like to use?'
        runtime = _get_choice_from_options(runtime, runtime_options, question, 'Runtime')
    if base_image:
        runtime = _get_runtime_from_image(base_image)
        if runtime is None:
            raise InvalidInitOptionException(f'Runtime could not be inferred for base image {base_image}.')
    package_types_options = runtime_options.get(runtime)
    if not package_types_options:
        raise InvalidInitOptionException(f'Lambda Runtime {runtime} is not supported for {use_case} examples.')
    if not pt_explicit:
        message = 'What package type would you like to use?'
        package_type = _get_choice_from_options(None, package_types_options, message, 'Package type')
        if package_type == IMAGE:
            base_image = _get_image_from_runtime(runtime)
    dependency_manager_options = package_types_options.get(package_type)
    if not dependency_manager_options:
        raise InvalidInitOptionException(f'{package_type} package type is not supported for {use_case} examples and runtime {runtime} selected.')
    dependency_manager = _get_dependency_manager(dependency_manager_options, dependency_manager, runtime)
    template_chosen = _get_app_template_choice(dependency_manager_options, dependency_manager)
    return (runtime, base_image, package_type, dependency_manager, template_chosen)

def prompt_user_to_enable_tracing():
    if False:
        for i in range(10):
            print('nop')
    '\n    Prompt user to if X-Ray Tracing should activated for functions in the SAM template and vice versa\n    '
    if click.confirm('\nWould you like to enable X-Ray tracing on the function(s) in your application? '):
        doc_link = 'https://aws.amazon.com/xray/pricing/'
        click.echo(f'X-Ray will incur an additional cost. View {doc_link} for more details')
        return True
    return False

def prompt_user_to_enable_application_insights():
    if False:
        i = 10
        return i + 15
    '\n    Prompt user to choose if AppInsights monitoring should be enabled for their application and vice versa\n    '
    doc_link = 'https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch-application-insights.html'
    if click.confirm(f'\nWould you like to enable monitoring using CloudWatch Application Insights?\nFor more info, please view {doc_link}'):
        pricing_link = 'https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/appinsights-what-is.html#appinsights-pricing'
        click.echo(f'AppInsights monitoring may incur additional cost. View {pricing_link} for more details')
        return True
    return False

def _get_choice_from_options(chosen, options, question, msg):
    if False:
        print('Hello World!')
    if chosen:
        return chosen
    click_choices = []
    options_list = options if isinstance(options, list) else list(options.keys())
    options_list = get_sorted_runtimes(options_list) if msg == 'Runtime' else options_list
    if not options_list:
        raise InvalidInitOptionException(f'There are no {msg} options available to be selected.')
    if len(options_list) == 1:
        click.echo(f'\nBased on your selections, the only {msg} available is {options_list[0]}.' + f'\nWe will proceed to selecting the {msg} as {options_list[0]}.')
        return options_list[0]
    click.echo(f'\n{question}')
    for (index, option) in enumerate(options_list):
        click.echo(f'\t{index + 1} - {option}')
        click_choices.append(str(index + 1))
    choice = click.prompt(msg, type=click.Choice(click_choices), show_choices=False)
    return options_list[int(choice) - 1]

def _get_app_template_choice(templates_options, dependency_manager):
    if False:
        return 10
    templates = _get_templates_with_dependency_manager(templates_options, dependency_manager)
    chosen_template = templates[0]
    if len(templates) > 1:
        click.echo('\nSelect your starter template')
        click_template_choices = []
        for (index, template) in enumerate(templates):
            click.echo(f"\t{index + 1} - {template['displayName']}")
            click_template_choices.append(str(index + 1))
        template_choice = click.prompt('Template', type=click.Choice(click_template_choices), show_choices=False)
        chosen_template = templates[int(template_choice) - 1]
    return chosen_template

def _get_dependency_manager(options, dependency_manager, runtime):
    if False:
        while True:
            i = 10
    valid_dep_managers = sorted(list(set((template['dependencyManager'] for template in options))))
    if not dependency_manager:
        if len(valid_dep_managers) == 1:
            dependency_manager = valid_dep_managers[0]
            click.echo(f'\nBased on your selections, the only dependency manager available is {dependency_manager}.' + f'\nWe will proceed copying the template using {dependency_manager}.')
        else:
            question = 'Which dependency manager would you like to use?'
            dependency_manager = _get_choice_from_options(dependency_manager, valid_dep_managers, question, 'Dependency manager')
    elif dependency_manager and dependency_manager not in valid_dep_managers:
        msg = f'Lambda Runtime {runtime} and dependency manager {dependency_manager} ' + 'do not have an available initialization template.'
        raise InvalidInitTemplateError(msg)
    return dependency_manager

def _get_schema_template_details(schemas_api_caller):
    if False:
        return 10
    try:
        return get_schema_template_details(schemas_api_caller)
    except ClientError as e:
        raise SchemasApiException('Exception occurs while getting Schemas template parameter. %s' % e.response['Error']['Message']) from e

def _package_schemas_code(runtime, schemas_api_caller, schema_template_details, output_dir, name, location):
    if False:
        while True:
            i = 10
    try:
        click.echo('Trying to get package schema code')
        with tempfile.NamedTemporaryFile(delete=False) as download_location:
            do_download_source_code_binding(runtime, schema_template_details, schemas_api_caller, download_location)
            do_extract_and_merge_schemas_code(download_location, output_dir, name, location)
    except (ClientError, WaiterError) as e:
        raise SchemasApiException('Exception occurs while packaging Schemas code. %s' % e.response['Error']['Message']) from e
    finally:
        remove(download_location.name)

def generate_summary_message(package_type, runtime, base_image, dependency_manager, output_dir, name, app_template, architecture):
    if False:
        for i in range(10):
            print('nop')
    "\n    Parameters\n    ----------\n    package_type : str\n        The package type, 'Zip' or 'Image', see samcli/lib/utils/packagetype.py\n    runtime : str\n        AWS Lambda function runtime\n    base_image : str\n        base image\n    dependency_manager : str\n        dependency manager\n    output_dir : str\n        the directory where project will be generated in\n    name : str\n        Project Name\n    app_template : str\n        application template generated\n    architecture : list\n        Architecture type either x86_64 or arm64 on AWS lambda\n\n    Returns\n    -------\n    str\n        Summary Message of the application template generated\n    "
    summary_msg = ''
    if package_type == ZIP:
        summary_msg = f"\n    -----------------------\n    Generating application:\n    -----------------------\n    Name: {name}\n    Runtime: {runtime}\n    Architectures: {architecture[0]}\n    Dependency Manager: {dependency_manager}\n    Application Template: {app_template}\n    Output Directory: {output_dir}\n    Configuration file: {pathlib.Path(output_dir).joinpath(name, DEFAULT_CONFIG_FILE_NAME)}\n    \n    Next steps can be found in the README file at {pathlib.Path(output_dir).joinpath(name, 'README.md')}\n        "
    elif package_type == IMAGE:
        summary_msg = f"\n    -----------------------\n    Generating application:\n    -----------------------\n    Name: {name}\n    Base Image: {base_image}\n    Architectures: {architecture[0]}\n    Dependency Manager: {dependency_manager}\n    Output Directory: {output_dir}\n    Configuration file: {pathlib.Path(output_dir).joinpath(name, DEFAULT_CONFIG_FILE_NAME)}\n\n    Next steps can be found in the README file at {pathlib.Path(output_dir).joinpath(name, 'README.md')}\n    "
    return summary_msg