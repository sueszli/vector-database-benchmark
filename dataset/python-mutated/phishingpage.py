"""
This module handles all the phishing related operations for
Wifiphisher.py
"""
import os
from shutil import copyfile
import wifiphisher.common.constants as constants
try:
    from configparser import ConfigParser, RawConfigParser
except ImportError:
    from configparser import ConfigParser, RawConfigParser

def config_section_map(config_file, section):
    if False:
        while True:
            i = 10
    '\n    Map the values of a config file to a dictionary.\n    '
    config = ConfigParser()
    config.read(config_file)
    dict1 = {}
    if section not in config.sections():
        return dict1
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
        except KeyError:
            dict1[option] = None
    return dict1

class InvalidTemplate(Exception):
    """ Exception class to raise in case of a invalid template """

    def __init__(self):
        if False:
            return 10
        Exception.__init__(self, 'The given template is either invalid or ' + 'not available locally!')

class PhishingTemplate(object):
    """ This class represents phishing templates """

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Construct object.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: None\n        :rtype: None\n        .. todo:: Maybe add a category field\n        '
        config_path = os.path.join(constants.phishing_pages_dir, name, 'config.ini')
        info = config_section_map(config_path, 'info')
        self._name = name
        self._display_name = info['name']
        self._description = info['description']
        self._payload = False
        self._config_path = os.path.join(constants.phishing_pages_dir, self._name, 'config.ini')
        if 'payloadpath' in info:
            self._payload = info['payloadpath']
        self._path = os.path.join(constants.phishing_pages_dir, self._name.lower(), constants.SCENARIO_HTML_DIR)
        self._path_static = os.path.join(constants.phishing_pages_dir, self._name.lower(), constants.SCENARIO_HTML_DIR, 'static')
        self._context = config_section_map(config_path, 'context')
        self._extra_files = []

    @staticmethod
    def update_config_file(payload_filename, config_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the configuration file\n\n        :param self: A PhishingTemplate object\n        :param payload_filename: the filename for the payload\n        :param config_path: the file path for the configuration\n        :type self: PhishingTemplate\n        :type payload_filename: str\n        :type config_path: str\n        :return: None\n        :rtype: None\n        '
        original_config = ConfigParser()
        original_config.read(config_path)
        config = RawConfigParser()
        config.add_section('info')
        options = original_config.options('info')
        for option in options:
            if option != 'payloadpath':
                config.set('info', option, original_config.get('info', option))
            else:
                dirname = os.path.dirname(original_config.get('info', 'payloadpath'))
                filepath = os.path.join(dirname, payload_filename)
                config.set('info', option, filepath)
        config.add_section('context')
        dirname = os.path.dirname(original_config.get('context', 'update_path'))
        filepath = os.path.join(dirname, payload_filename)
        config.set('context', 'update_path', filepath)
        with open(config_path, 'w') as configfile:
            config.write(configfile)

    def update_payload_path(self, filename):
        if False:
            print('Hello World!')
        '\n        :param self: A PhishingTemplate object\n        :filename: the filename for the payload\n        :type self: PhishingTemplate\n        :type filename: str\n        :return: None\n        :rtype: None\n        '
        config_path = self._config_path
        self.update_config_file(filename, config_path)
        info = config_section_map(config_path, 'info')
        self._payload = False
        if 'payloadpath' in info:
            self._payload = info['payloadpath']
        self._context = config_section_map(config_path, 'context')
        self._extra_files = []

    def merge_context(self, context):
        if False:
            print('Hello World!')
        '\n            Merge dict context with current one\n            In case of confict always keep current values\n        '
        context.update(self._context)
        self._context = context

    def get_context(self):
        if False:
            while True:
                i = 10
        '\n        Return the context of the template.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: the context of the template\n        :rtype: dict\n        '
        return self._context

    def get_display_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the display name of the template.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: the display name of the template\n        :rtype: str\n        '
        return self._display_name

    def get_payload_path(self):
        if False:
            return 10
        '\n        Return the payload path of the template.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: The path of the template\n        :rtype: bool\n        '
        return self._payload

    def has_payload(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return whether the template has a payload.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: boolean if it needs payload\n        :rtype: bool\n        '
        if self._payload:
            return True
        return False

    def get_description(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the description of the template.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: the description of the template\n        :rtype: str\n        '
        return self._description

    def get_path(self):
        if False:
            while True:
                i = 10
        '\n        Return the path of the template files.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: the path of template files\n        :rtype: str\n        '
        return self._path

    def get_path_static(self):
        if False:
            return 10
        '\n        Return the path of the static template files.\n        JS, CSS, Image files lie there.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: the path of static template files\n        :rtype: str\n        '
        return self._path_static

    def use_file(self, path):
        if False:
            while True:
                i = 10
        '\n        Copies a file in the filesystem to the path\n        of the template files.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :param path: path of the file that is to be copied\n        :type self: str\n        :return: the path of the file under the template files\n        :rtype: str\n        '
        if path is not None and os.path.isfile(path):
            filename = os.path.basename(path)
            copyfile(path, self.get_path_static() + filename)
            self._extra_files.append(self.get_path_static() + filename)
            return filename

    def remove_extra_files(self):
        if False:
            print('Hello World!')
        '\n        Removes extra used files (if any)\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: None\n        :rtype: None\n        '
        for filename in self._extra_files:
            if os.path.isfile(filename):
                os.remove(filename)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string representation of the template.\n\n        :param self: A PhishingTemplate object\n        :type self: PhishingTemplate\n        :return: the name followed by the description of the template\n        :rtype: str\n        '
        return self._display_name + '\n\t' + self._description + '\n'

class TemplateManager(object):
    """ This class handles all the template management operations """

    def __init__(self, data_pages=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct object.\n\n        :param self: A TemplateManager object\n        :param data_pages: The directory containing the templates\n        :type self: TemplateManager\n        :return: None\n        :rtype: None\n        '
        self._template_directory = data_pages or constants.phishing_pages_dir
        if data_pages:
            constants.phishing_pages_dir = data_pages
        page_dirs = os.listdir(self._template_directory)
        self._templates = {}
        for page in page_dirs:
            if os.path.isdir(page) and self.is_valid_template(page)[0]:
                self._templates[page] = PhishingTemplate(page)
        self.add_user_templates()

    def get_templates(self):
        if False:
            while True:
                i = 10
        '\n        Return all the available templates.\n\n        :param self: A TemplateManager object\n        :type self: TemplateManager\n        :return: all the available templates\n        :rtype: dict\n        '
        return self._templates

    def is_valid_template(self, name):
        if False:
            while True:
                i = 10
        '\n        Validate the template\n        :param self: A TemplateManager object\n        :param name: A directory name\n        :type self: A TemplateManager object\n        :return: tuple of is_valid and output string\n        :rtype: tuple\n        '
        html = False
        dir_path = os.path.join(self._template_directory, name)
        if not 'config.ini' in os.listdir(dir_path):
            return (False, 'Configuration file not found in: ')
        try:
            tdir = os.listdir(os.path.join(dir_path, constants.SCENARIO_HTML_DIR))
        except OSError:
            return (False, 'No ' + constants.SCENARIO_HTML_DIR + ' directory found in: ')
        for tfile in tdir:
            if tfile.endswith('.html'):
                html = True
                break
        if not html:
            return (False, 'No HTML files found in: ')
        return (True, name)

    def find_user_templates(self):
        if False:
            i = 10
            return i + 15
        "\n        Return all the user's templates available.\n\n        :param self: A TemplateManager object\n        :type self: TemplateManager\n        :return: all the local templates available\n        :rtype: list\n        "
        local_templates = []
        for name in os.listdir(self._template_directory):
            if os.path.isdir(os.path.join(self._template_directory, name)) and name not in self._templates:
                (is_valid, output) = self.is_valid_template(name)
                if is_valid:
                    local_templates.append(name)
                else:
                    print('[' + constants.R + '!' + constants.W + '] ' + output + name)
        return local_templates

    def add_user_templates(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add all the user templates to the database.\n\n        :param self: A TemplateManager object\n        :type: self: TemplateManager\n        :return: None\n        :rtype: None\n        '
        user_templates = self.find_user_templates()
        for template in user_templates:
            local_template = PhishingTemplate(template)
            self._templates[template] = local_template

    @property
    def template_directory(self):
        if False:
            for i in range(10):
                print('nop')
        return self._template_directory

    def on_exit(self):
        if False:
            return 10
        '\n        Delete any extra files on exit\n\n        :param self: A TemplateManager object\n        :type: self: TemplateManager\n        :return: None\n        :rtype: None\n        '
        for templ_obj in list(self._templates.values()):
            templ_obj.remove_extra_files()