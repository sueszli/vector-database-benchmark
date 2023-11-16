"""
Class used to parse and update template when tracing is enabled
"""
import logging
from typing import Any
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.representer import RoundTripRepresenter
from samcli.lib.init.template_modifiers.cli_template_modifier import TemplateModifier
LOG = logging.getLogger(__name__)

class XRayTracingTemplateModifier(TemplateModifier):
    FIELD_NAME_FUNCTION_TRACING = 'Tracing'
    FIELD_NAME_API_TRACING = 'TracingEnabled'
    GLOBALS = 'Globals'
    RESOURCE = 'Resources'
    FUNCTION = 'Function'
    TRACING_FUNCTION = 'Tracing'
    ACTIVE_TRACING = 'Active'
    API = 'Api'
    TRACING_API = 'TracingEnabled'
    TRACING_API_VALUE = True
    COMMENT = '# More info about Globals: https://github.com/awslabs/serverless-application-model/blob/master/docs/globals.rst\n'

    class NonAliasingRTRepresenter(RoundTripRepresenter):

        def ignore_aliases(self, data):
            if False:
                return 10
            return True

    def __init__(self, location):
        if False:
            for i in range(10):
                print('nop')
        self.yaml = YAML()
        self.yaml.Representer = XRayTracingTemplateModifier.NonAliasingRTRepresenter
        super().__init__(location)

    def _get_template(self) -> Any:
        if False:
            i = 10
            return i + 15
        with open(self.template_location) as file:
            return self.yaml.load(file)

    def _update_template_fields(self):
        if False:
            i = 10
            return i + 15
        '\n        Add new field to SAM template\n        '
        if self.template.get(self.GLOBALS):
            template_globals = self.template.get(self.GLOBALS)
            function_globals = template_globals.get(self.FUNCTION, {})
            if not function_globals:
                template_globals[self.FUNCTION] = {}
            template_globals[self.FUNCTION][self.TRACING_FUNCTION] = self.ACTIVE_TRACING
            api_globals = template_globals.get(self.API, {})
            if not api_globals:
                template_globals[self.API] = {}
            template_globals[self.API][self.TRACING_API] = self.TRACING_API_VALUE
        else:
            self._add_tracing_with_globals()

    def _add_tracing_with_globals(self):
        if False:
            for i in range(10):
                print('nop')
        'Adds Globals and tracing fields'
        global_section = {self.FUNCTION: {self.TRACING_FUNCTION: self.ACTIVE_TRACING}, self.API: {self.TRACING_API: self.TRACING_API_VALUE}}
        self.template = CommentedMap(self.template)
        self.template[self.GLOBALS] = CommentedMap(global_section)
        self.template.yaml_set_comment_before_after_key(self.GLOBALS, before=self.COMMENT)

    def _print_sanity_check_error(self):
        if False:
            return 10
        link = 'https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-resource-function.html#sam-function-tracing'
        message = f'Warning: Unable to add Tracing to the project. To learn more about Tracing visit {link}'
        LOG.warning(message)

    def _write(self, template: list):
        if False:
            i = 10
            return i + 15
        '\n        write generated template into SAM template\n\n        Parameters\n        ----------\n        template : list\n            array with updated template data\n        '
        with open(self.template_location, 'w') as file:
            self.yaml.dump(self.template, file)