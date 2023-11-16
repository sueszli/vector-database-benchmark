import json
import re
from typing import Any, Dict, List, Mapping, Sequence
import markdown
from django.utils.html import escape as escape_html
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITES
from zerver.openapi.openapi import check_deprecated_consistency, get_openapi_parameters, get_parameters_description
REGEXP = re.compile('\\{generate_api_arguments_table\\|\\s*(.+?)\\s*\\|\\s*(.+)\\s*\\}')
API_PARAMETER_TEMPLATE = '\n<div class="api-argument" id="parameter-{argument}">\n    <p class="api-argument-name"><strong>{argument}</strong> <span class="api-field-type">{type}</span> {required} {deprecated} <a href="#parameter-{argument}" class="api-argument-hover-link"><i class="fa fa-chain"></i></a></p>\n    <div class="api-example">\n        <span class="api-argument-example-label">Example</span>: <code>{example}</code>\n    </div>\n    <div class="api-description">{description}{object_details}</div>\n    <hr>\n</div>\n'.strip()
OBJECT_DETAILS_TEMPLATE = '\n<p><strong>{argument}</strong> object details:</p>\n<ul>\n{values}\n</ul>\n'.strip()
OBJECT_LIST_ITEM_TEMPLATE = '\n<li>\n<code>{value}</code>: <span class=api-field-type>{data_type}</span> {required} {description}{object_details}\n</li>\n'.strip()
OBJECT_DESCRIPTION_TEMPLATE = '\n{description}\n<p>{additional_information}</p>\n'.strip()
OBJECT_CODE_TEMPLATE = '<code>{value}</code>'.strip()

class MarkdownArgumentsTableGenerator(Extension):

    @override
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        if False:
            print('Hello World!')
        md.preprocessors.register(APIArgumentsTablePreprocessor(md, self.getConfigs()), 'generate_api_arguments', PREPROCESSOR_PRIORITES['generate_api_arguments'])

class APIArgumentsTablePreprocessor(Preprocessor):

    def __init__(self, md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        if False:
            return 10
        super().__init__(md)

    @override
    def run(self, lines: List[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        done = False
        while not done:
            for line in lines:
                loc = lines.index(line)
                match = REGEXP.search(line)
                if not match:
                    continue
                doc_name = match.group(2)
                (endpoint, method) = doc_name.rsplit(':', 1)
                arguments: List[Dict[str, Any]] = []
                try:
                    arguments = get_openapi_parameters(endpoint, method)
                except KeyError as e:
                    if e.args != ('parameters',):
                        raise e
                if arguments:
                    text = self.render_parameters(arguments)
                elif get_parameters_description(endpoint, method) == '':
                    text = ['This endpoint does not accept any parameters.']
                else:
                    text = []
                line_split = REGEXP.split(line, maxsplit=0)
                preceding = line_split[0]
                following = line_split[-1]
                text = [preceding, *text, following]
                lines = lines[:loc] + text + lines[loc + 1:]
                break
            else:
                done = True
        return lines

    def render_parameters(self, arguments: Sequence[Mapping[str, Any]]) -> List[str]:
        if False:
            print('Hello World!')
        parameters = []
        md_engine = markdown.Markdown(extensions=[])
        arguments = sorted(arguments, key=lambda argument: 'deprecated' in argument)
        for argument in arguments:
            name = argument.get('argument') or argument.get('name')
            description = argument['description']
            enums = argument.get('schema', {}).get('enum')
            if enums is not None:
                formatted_enums = [OBJECT_CODE_TEMPLATE.format(value=json.dumps(enum)) for enum in enums]
                description += '\nMust be one of: {}. '.format(', '.join(formatted_enums))
            default = argument.get('schema', {}).get('default')
            if default is not None:
                description += f'\nDefaults to `{json.dumps(default)}`.'
            data_type = ''
            if 'schema' in argument:
                data_type = generate_data_type(argument['schema'])
            else:
                data_type = generate_data_type(argument['content']['application/json']['schema'])
            example = ''
            if 'example' in argument:
                example = json.dumps(argument['example'])
            else:
                example = json.dumps(argument['content']['application/json']['example'])
            required_string: str = 'required'
            if argument.get('in', '') == 'path':
                assert argument['required']
                required_string = 'required in path'
            if argument.get('required', False):
                required_block = f'<span class="api-argument-required">{required_string}</span>'
            else:
                required_block = '<span class="api-argument-optional">optional</span>'
            check_deprecated_consistency(argument, description)
            if argument.get('deprecated', False):
                deprecated_block = '<span class="api-argument-deprecated">Deprecated</span>'
            else:
                deprecated_block = ''
            object_block = ''
            if 'object' in data_type:
                if 'schema' in argument:
                    object_schema = argument['schema']
                else:
                    object_schema = argument['content']['application/json']['schema']
                if 'items' in object_schema and 'properties' in object_schema['items']:
                    object_block = self.render_object_details(object_schema['items'], str(name))
                elif 'properties' in object_schema:
                    object_block = self.render_object_details(object_schema, str(name))
            parameters.append(API_PARAMETER_TEMPLATE.format(argument=name, example=escape_html(example), required=required_block, deprecated=deprecated_block, description=md_engine.convert(description), type=data_type, object_details=object_block))
        return parameters

    def render_object_details(self, schema: Mapping[str, Any], name: str) -> str:
        if False:
            i = 10
            return i + 15
        md_engine = markdown.Markdown(extensions=[])
        li_elements = []
        object_values = schema.get('properties', {})
        for value in object_values:
            description = ''
            if 'description' in object_values[value]:
                description = object_values[value]['description']
            additions: List[str] = []
            default = object_values.get(value, {}).get('default')
            if default is not None:
                formatted_default = OBJECT_CODE_TEMPLATE.format(value=json.dumps(default))
                additions += f'Defaults to {formatted_default}. '
            enums = object_values.get(value, {}).get('enum')
            if enums is not None:
                formatted_enums = [OBJECT_CODE_TEMPLATE.format(value=json.dumps(enum)) for enum in enums]
                additions += 'Must be one of: {}. '.format(', '.join(formatted_enums))
            if 'example' in object_values[value]:
                example = json.dumps(object_values[value]['example'])
                formatted_example = OBJECT_CODE_TEMPLATE.format(value=escape_html(example))
                additions += f'<span class="api-argument-example-label">Example</span>: {formatted_example}'
            if len(additions) > 0:
                additional_information = ''.join(additions).strip()
                description_final = OBJECT_DESCRIPTION_TEMPLATE.format(description=md_engine.convert(description), additional_information=additional_information)
            else:
                description_final = md_engine.convert(description)
            required_block = ''
            if 'required' in schema:
                if value in schema['required']:
                    required_block = '<span class="api-argument-required">required</span>'
                else:
                    required_block = '<span class="api-argument-optional">optional</span>'
            data_type = generate_data_type(object_values[value])
            details = ''
            if 'object' in data_type and 'properties' in object_values[value]:
                details += self.render_object_details(object_values[value], str(value))
            li = OBJECT_LIST_ITEM_TEMPLATE.format(value=value, data_type=data_type, required=required_block, description=description_final, object_details=details)
            li_elements.append(li)
        object_details = OBJECT_DETAILS_TEMPLATE.format(argument=name, values='\n'.join(li_elements))
        return object_details

def makeExtension(*args: Any, **kwargs: str) -> MarkdownArgumentsTableGenerator:
    if False:
        print('Hello World!')
    return MarkdownArgumentsTableGenerator(*args, **kwargs)

def generate_data_type(schema: Mapping[str, Any]) -> str:
    if False:
        while True:
            i = 10
    data_type = ''
    if 'oneOf' in schema:
        for item in schema['oneOf']:
            data_type = data_type + generate_data_type(item) + ' | '
        data_type = data_type[:-3]
    elif 'items' in schema:
        data_type = '(' + generate_data_type(schema['items']) + ')[]'
    else:
        data_type = schema['type']
        if 'nullable' in schema and schema['nullable']:
            data_type = data_type + ' | null'
    return data_type