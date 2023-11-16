import os
import frontmatter
import shutil
import tempfile
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound, meta
from lwe.core.config import Config
from lwe.core.logger import Logger
import lwe.core.util as util
TEMP_TEMPLATE_DIR = 'lwe-temp-templates'

class TemplateManager:
    """
    Manage templates.
    """

    def __init__(self, config=None):
        if False:
            print('Hello World!')
        '\n        Initializes the class with the given configuration.\n\n        :param config: Configuration settings. If not provided, a default Config object is used.\n        :type config: Config, optional\n        '
        self.config = config or Config()
        self.log = Logger(self.__class__.__name__, self.config)
        self.temp_template_dir = self.make_temp_template_dir()
        self.user_template_dirs = self.config.args.template_dir or util.get_environment_variable_list('template_dir') or self.config.get('directories.templates')
        self.make_user_template_dirs()
        self.system_template_dirs = [os.path.join(util.get_package_root(self), 'templates')]
        self.all_template_dirs = self.user_template_dirs + self.system_template_dirs + [self.temp_template_dir]
        self.templates = []
        self.templates_env = None

    def template_builtin_variables(self):
        if False:
            return 10
        '\n        This method returns a dictionary of built-in variables.\n\n        :return: A dictionary where the key is the variable name and the value is the function associated with it.\n        :rtype: dict\n        '
        return {'clipboard': util.paste_from_clipboard}

    def ensure_template(self, template_name):
        if False:
            print('Hello World!')
        '\n        Checks if a template exists.\n\n        :param template_name: The name of the template to check.\n        :type template_name: str\n        :return: A tuple containing a boolean indicating if the template exists, the template name, and a message.\n        :rtype: tuple\n        '
        if not template_name:
            return (False, None, 'No template name specified')
        self.log.debug(f'Ensuring template {template_name} exists')
        self.load_templates()
        if template_name not in self.templates:
            return (False, template_name, f'Template {template_name!r} not found')
        message = f'Template {template_name} exists'
        self.log.debug(message)
        return (True, template_name, message)

    def get_template_variables_substitutions(self, template_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get template variables and their substitutions.\n\n        :param template_name: The name of the template\n        :type template_name: str\n        :return: A tuple containing a boolean indicating success, the template with its variables and substitutions, and a user message\n        :rtype: tuple\n        '
        (success, template_name, user_message) = self.ensure_template(template_name)
        if not success:
            return (success, template_name, user_message)
        (template, variables) = self.get_template_and_variables(template_name)
        substitutions = self.process_template_builtin_variables(template_name, variables)
        return (True, (template, variables, substitutions), f'Loaded template substitutions: {template_name}')

    def render_template(self, template_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Render a template with variable substitutions.\n\n        :param template_name: The name of the template to render\n        :type template_name: str\n        :return: A tuple containing a success flag, the rendered message or template name, and a user message\n        :rtype: tuple\n        '
        (success, response, user_message) = self.get_template_variables_substitutions(template_name)
        if not success:
            return (success, template_name, user_message)
        (template, variables, substitutions) = response
        message = template.render(**substitutions)
        return (True, message, f'Rendered template: {template_name}')

    def get_template_source(self, template_name):
        if False:
            while True:
                i = 10
        '\n        Get the source of a specified template.\n\n        :param template_name: The name of the template\n        :type template_name: str\n        :return: A tuple containing a boolean indicating success, the source of the template if successful, and a user message\n        :rtype: tuple\n        '
        (success, template_name, user_message) = self.ensure_template(template_name)
        if not success:
            return (success, template_name, user_message)
        (template, _) = self.get_template_and_variables(template_name)
        source = frontmatter.load(template.filename)
        return (True, source, f'Loaded template source: {template_name}')

    def get_template_editable_filepath(self, template_name):
        if False:
            print('Hello World!')
        '\n        Get the editable file path for a given template.\n\n        :param template_name: The name of the template\n        :type template_name: str\n        :return: A tuple containing a boolean indicating if the template is editable, the file path of the template, and a message\n        :rtype: tuple\n        '
        if not template_name:
            return (False, template_name, 'No template name specified')
        (template, _) = self.get_template_and_variables(template_name)
        if template:
            filename = template.filename
            if self.is_system_template(filename):
                return (False, template_name, f'{template_name} is a system template, and cannot be edited directly')
        else:
            filename = os.path.join(self.user_template_dirs[0], template_name)
        return (True, filename, f'Template {filename} can be edited')

    def copy_template(self, old_name, new_name):
        if False:
            i = 10
            return i + 15
        '\n        Copies a template file to a new location.\n\n        :param old_name: The name of the existing template file.\n        :type old_name: str\n        :param new_name: The name for the new template file.\n        :type new_name: str\n        :return: A tuple containing a boolean indicating success or failure, the new file path, and a status message.\n        :rtype: tuple\n        '
        (template, _) = self.get_template_and_variables(old_name)
        if not template:
            return (False, old_name, f'{old_name} does not exist')
        old_filepath = template.filename
        base_filepath = self.user_template_dirs[0] if self.is_system_template(old_filepath) else os.path.dirname(old_filepath)
        new_filepath = os.path.join(base_filepath, new_name)
        if os.path.exists(new_filepath):
            return (False, new_filepath, f'{new_filepath} already exists')
        shutil.copy2(old_filepath, new_filepath)
        self.load_templates()
        return (True, new_filepath, f'Copied template {old_filepath} to {new_filepath}')

    def template_can_delete(self, template_name):
        if False:
            print('Hello World!')
        '\n        Checks if a template can be deleted.\n\n        :param template_name: The name of the template to check\n        :type template_name: str\n        :return: A tuple containing a boolean indicating if the template can be deleted, the template name or filename, and a message\n        :rtype: tuple\n        '
        if not template_name:
            return (False, template_name, 'No template name specified')
        (template, _) = self.get_template_and_variables(template_name)
        if template:
            filename = template.filename
            if self.is_system_template(filename):
                return (False, filename, f'{filename} is a system template, and cannot be deleted')
        else:
            return (False, template_name, f'{template_name} does not exist')
        return (True, filename, f'Template {filename} can be deleted')

    def template_delete(self, filename):
        if False:
            i = 10
            return i + 15
        '\n        Deletes a specified template file and reloads the templates.\n\n        :param filename: The name of the file to be deleted.\n        :type filename: str\n        :return: A tuple containing a boolean indicating success, the filename, and a message.\n        :rtype: tuple\n        '
        os.remove(filename)
        self.load_templates()
        return (True, filename, f'Deleted {filename}')

    def extract_metadata_keys(self, keys, metadata):
        if False:
            for i in range(10):
                print('nop')
        '\n        Extracts specified keys from the metadata.\n\n        :param keys: Keys to be extracted from the metadata.\n        :type keys: list\n        :param metadata: The metadata from which keys are to be extracted.\n        :type metadata: dict\n        :return: A tuple containing the updated metadata and the extracted keys.\n        :rtype: tuple\n        '
        extracted_keys = {}
        for key in keys:
            if key in metadata:
                extracted_keys[key] = metadata[key]
                del metadata[key]
        return (metadata, extracted_keys)

    def extract_template_run_overrides(self, metadata):
        if False:
            i = 10
            return i + 15
        '\n        Extracts template run overrides from metadata.\n\n        :param metadata: The metadata from which to extract overrides.\n        :type metadata: dict\n        :return: A tuple containing the updated metadata and the extracted overrides.\n        :rtype: tuple\n        '
        override_keys = ['request_overrides']
        builtin_keys = ['description']
        (metadata, overrides) = self.extract_metadata_keys(override_keys, metadata)
        (metadata, _) = self.extract_metadata_keys(builtin_keys, metadata)
        return (metadata, overrides)

    def build_message_from_template(self, template_name, substitutions=None):
        if False:
            return 10
        '\n        Build a message from a given template and substitutions.\n\n        :param template_name: The name of the template to use.\n        :type template_name: str\n        :param substitutions: The substitutions to apply to the template. Defaults to None.\n        :type substitutions: dict, optional\n        :return: The rendered message and any overrides.\n        :rtype: tuple\n        '
        substitutions = substitutions or {}
        (template, _) = self.get_template_and_variables(template_name)
        source = frontmatter.load(template.filename)
        (template_substitutions, overrides) = self.extract_template_run_overrides(source.metadata)
        final_substitutions = {**template_substitutions, **substitutions}
        self.log.debug(f'Rendering template: {template_name}')
        final_template = Template(source.content)
        message = final_template.render(**final_substitutions)
        return (message, overrides)

    def process_template_builtin_variables(self, template_name, variables=None):
        if False:
            return 10
        '\n        Process the built-in variables in a template.\n\n        :param template_name: The name of the template\n        :type template_name: str\n        :param variables: The variables to be processed, defaults to None\n        :type variables: list, optional\n        :return: A dictionary of substitutions for the variables\n        :rtype: dict\n        '
        variables = variables or []
        builtin_variables = self.template_builtin_variables()
        substitutions = {}
        for (variable, method) in builtin_variables.items():
            if variable in variables:
                substitutions[variable] = method()
                self.log.debug(f'Collected builtin variable {variable} for template {template_name}: {substitutions[variable]}')
        return substitutions

    def make_user_template_dirs(self):
        if False:
            i = 10
            return i + 15
        '\n        Create directories for user templates if they do not exist.\n\n        :return: None\n        '
        for template_dir in self.user_template_dirs:
            if not os.path.exists(template_dir):
                os.makedirs(template_dir)

    def make_temp_template_dir(self):
        if False:
            while True:
                i = 10
        '\n        Create directory for temporary templates if it does not exist.\n\n        :return: None\n        '
        temp_dir = os.path.join(tempfile.gettempdir(), TEMP_TEMPLATE_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        util.clean_directory(temp_dir)
        return temp_dir

    def make_temp_template(self, template_contents, suffix='md'):
        if False:
            i = 10
            return i + 15
        "\n        Create a temporary template.\n\n        :param template_contents: The contents to be written to the temporary template\n        :type template_contents: str\n        :param suffix: The suffix for the temporary file, defaults to 'md'\n        :type suffix: str, optional\n        :return: The basename and the full path of the temporary template\n        :rtype: tuple\n        "
        filepath = util.write_temp_file(template_contents, suffix='md', dir=self.temp_template_dir)
        return (os.path.basename(filepath), filepath)

    def remove_temp_template(self, template_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove a temporary template.\n\n        :param template_name: The name of the temporary template\n        :type template_name: str\n        :return: None\n        '
        filepath = os.path.join(self.temp_template_dir, template_name)
        if os.path.exists(filepath):
            os.remove(filepath)

    def load_templates(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load templates from directories.\n\n        :return: None\n        '
        self.log.debug('Loading templates from dirs: %s' % ', '.join(self.all_template_dirs))
        jinja_env = Environment(loader=FileSystemLoader(self.all_template_dirs))
        filenames = jinja_env.list_templates()
        self.templates_env = jinja_env
        self.templates = filenames or []

    def get_template_and_variables(self, template_name):
        if False:
            print('Hello World!')
        '\n        Fetches a template and its variables.\n\n        :param template_name: The name of the template to fetch\n        :type template_name: str\n        :return: The fetched template and its variables, or (None, None) if the template is not found\n        :rtype: tuple\n        '
        try:
            template = self.templates_env.get_template(template_name)
        except TemplateNotFound:
            return (None, None)
        template_source = self.templates_env.loader.get_source(self.templates_env, template_name)
        parsed_content = self.templates_env.parse(template_source)
        variables = meta.find_undeclared_variables(parsed_content)
        return (template, variables)

    def is_system_template(self, filepath):
        if False:
            i = 10
            return i + 15
        '\n        Check if a file is a system template.\n\n        :param filepath: The path of the file to check\n        :type filepath: str\n        :return: True if the file is a system template, False otherwise\n        :rtype: bool\n        '
        for dir in self.system_template_dirs:
            if filepath.startswith(dir):
                return True
        return False