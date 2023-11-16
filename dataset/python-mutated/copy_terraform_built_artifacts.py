"""
Python script that prepares artifacts for SAM CLI to invoke. 
This script will work in both Linux/MacOS and Windows.

It consists of 5 steps:
1. create a temporary TF backend (create_backend_override)
2. run `terraform init -reconfigure`
3. run `terraform apply` on the SAM CLI Metadata resource
4. run `terraform out` to produce an output
5. parse the output to locate the built artifact, and move it to the SAM CLI 
build artifact directory (find_and_copy_assets)

Note: This script intentionally does not use Python3 specific syntax.

"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from zip import unzip
LOG = logging.getLogger(__name__)
TF_BACKEND_OVERRIDE_FILENAME = 'z_samcli_backend_override'
TF_BLOCKED_ARGUMENTS = ['-target', '-destroy']
TF_ENVIRONMENT_VARIABLE_DELIM = '='
TF_ENVIRONMENT_VARIABLES = ['TF_CLI_ARGS', 'TF_CLI_ARGS_plan', 'TF_CLI_ARGS_apply']

class ResolverException(Exception):
    """
    Exception raised by resolver objects
    """

    def __init__(self, message):
        if False:
            while True:
                i = 10
        self.message = message
        super(ResolverException, self).__init__(self.message)

class Tokenizer(object):
    """
    Tokenizer class that specifies the delimiter it will be operating on,
    prior to applying to an input_string
    """

    def __init__(self, delimiter=None):
        if False:
            return 10
        self.delimiter = delimiter if delimiter else '|'

    def tokenize(self, input_string):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tokenizes an input string based on specified delimiter.\n        '
        return [token for token in input_string.split(self.delimiter) if token]

class Resolver(object):
    """
    Base Resolver class that exposes a `resolve` method.
    """

    def resolve(self, structured_object):
        if False:
            print('Hello World!')
        '\n        returns a portion of the structured_object based on the resolving rules applied.\n        '
        raise NotImplementedError

class KeyResolver(Resolver):
    """
    Resolver that operates on a dict structure and matches against a specified condition
    supplied as a key.
    """

    def __init__(self, key):
        if False:
            while True:
                i = 10
        self.key = key

    def resolve(self, structured_object):
        if False:
            for i in range(10):
                print('nop')
        if not structured_object or not isinstance(structured_object, dict):
            raise ResolverException('Data object malformed: {}'.format(structured_object))
        return structured_object.get(self.key, {})

class ListConditionResolver(Resolver):
    """
    Resolver that operates on a list structure and matches against a specified condition
    supplied as a key, value.
    """

    def __init__(self, key, value):
        if False:
            while True:
                i = 10
        self.key = key
        self.value = value.strip('"')

    def resolve(self, structured_object):
        if False:
            for i in range(10):
                print('nop')
        if not structured_object or not isinstance(structured_object, list):
            raise ResolverException('Data object malformed: {}'.format(structured_object))
        for item in structured_object:
            if isinstance(item, dict) and item.get(self.key) == self.value:
                return item
        return {}

class Parser(object):
    """
    Extremely simple parser that relies on a match against a regex
    to see if the given token needs to be mapped to
    a KeyResolver or a ListConditionResolver.
    """

    def __init__(self, expression):
        if False:
            i = 10
            return i + 15
        self.expression = expression
        self.resolvers = []
        self.list_resolver_regex = re.compile('\\[\\?(\\S+)==(\\S+)\\]')
        self.tokens = Tokenizer().tokenize(self.expression)
        for token in self.tokens:
            self.resolvers.append(self.find_resolver(token))

    def parse(self):
        if False:
            print('Hello World!')
        '\n        Instantiate a searcher that returns parsed data based on the resolvers.\n        :return:\n        '
        return Searcher(self.resolvers)

    def find_resolver(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the resolver for the appropriate token. The implementation of this function\n        is a direct match against a regex as the number of uses-case to be supported are less.\n        '
        groups = self.list_resolver_regex.findall(token)
        if not groups:
            return KeyResolver(key=token)
        else:
            return ListConditionResolver(key=groups[0][0], value=groups[0][1])

class Searcher(object):
    """
    Searcher class that allows for searching a Jpath against structured data.
    """

    def __init__(self, resolvers):
        if False:
            for i in range(10):
                print('nop')
        self.resolvers = resolvers

    def search(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Search by applying all resolvers against structured data.\n        '
        for resolver in self.resolvers:
            data = resolver.resolve(data)
        return data

def copytree(src, dst):
    if False:
        i = 10
        return i + 15
    'Modified copytree method\n    Note: before python3.8 there is no `dir_exists_ok` argument, therefore\n    this function explicitly creates one if it does not exist.\n    '
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

def cli_exit():
    if False:
        return 10
    '\n    Unsuccessful exit code for the script.\n    '
    sys.exit(1)

def find_and_copy_assets(directory_path, expression, data_object):
    if False:
        return 10
    '\n    Takes in an expression, directory_path and a json input from the standard input,\n    tries to find the appropriate element within the json based on the element. It then takes action to\n    either copy or unzip to the specified directory_path.\n\n    Parameters:\n    -----------\n    directory_path: str\n        path of the directory to move the built artifact to\n    expression: str\n        jpath-like expression to locate the original location of the built artifact\n    data_object: str/bytes\n        a json input, produced from `terraform out`\n    '
    directory_path = os.path.abspath(directory_path)
    terraform_project_root = os.getcwd()
    extracted_attribute_path = None
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        LOG.error('Expected --directory to be a valid directory!')
        cli_exit()
    try:
        extracted_attribute_path = Parser(expression=expression).parse().search(data=data_object)
    except ResolverException as ex:
        LOG.error(ex.message, exc_info=True)
        cli_exit()
    extracted_attribute_path = str(extracted_attribute_path)
    abs_attribute_path = os.path.abspath(os.path.join(terraform_project_root, extracted_attribute_path)) if not os.path.isabs(extracted_attribute_path) else os.path.abspath(extracted_attribute_path)
    if not os.path.exists(abs_attribute_path):
        LOG.error('Extracted attribute path from provided expression does not exist!')
        cli_exit()
    if abs_attribute_path == directory_path:
        LOG.error('Extracted expression path cannot be the same as the supplied directory path')
        cli_exit()
    try:
        if zipfile.is_zipfile(abs_attribute_path):
            unzip(abs_attribute_path, directory_path)
        else:
            copytree(abs_attribute_path, directory_path)
    except OSError as ex:
        LOG.error('Copy/Unzip unsuccessful!', exc_info=ex)
        cli_exit()

def create_backend_override():
    if False:
        return 10
    '\n    Copies and rename the override tf file from the metadata directory to the root\n    directory of the TF application.\n    '
    override_src_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), TF_BACKEND_OVERRIDE_FILENAME)
    override_dest_path = os.path.join(os.getcwd(), TF_BACKEND_OVERRIDE_FILENAME + '.tf')
    try:
        shutil.copy2(override_src_path, override_dest_path)
    except OSError as ex:
        LOG.error('Copy unsuccessful!', exc_info=ex)
        cli_exit()

def validate_environment_variables():
    if False:
        while True:
            i = 10
    '\n    Validate that the Terraform environment variables do not contain blocked arguments.\n    '
    for env_var in TF_ENVIRONMENT_VARIABLES:
        env_value = os.environ.get(env_var, '')
        trimmed_arguments = []
        for argument in env_value.split(' '):
            cleaned_argument = argument.strip()
            cleaned_argument = cleaned_argument.split(TF_ENVIRONMENT_VARIABLE_DELIM)[0]
            trimmed_arguments.append(cleaned_argument)
        if any([argument in TF_BLOCKED_ARGUMENTS for argument in trimmed_arguments]):
            LOG.error("Environment variable '%s' contains a blocked argument, please validate it does not contain: %s" % (env_var, TF_BLOCKED_ARGUMENTS))
            cli_exit()
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Copy built artifacts referenced in a json file (passed via stdin) matching a search pattern')
    argparser.add_argument('--expression', type=str, required=True, help='Jpath query expression separated by | (delimiter) and allows for searching within a json object.eg: |values|sub_module|[?address==me]|output')
    argparser.add_argument('--directory', type=str, required=True, help='Directory to which extracted expression contents are copied/unzipped to')
    argparser.add_argument('--target', type=str, required=False, help='Terraform resource path for the SAM CLI Metadata resource. This option is not to be used with --json')
    argparser.add_argument('--json', type=str, required=False, help='Terraform output json body. This option is not to be used with --target.')
    arguments = argparser.parse_args()
    directory_path = os.path.abspath(arguments.directory)
    expression = arguments.expression
    target = arguments.target
    json_str = arguments.json
    validate_environment_variables()
    if target and json_str:
        LOG.error('Provide either --target or --json. Do not provide both.')
        cli_exit()
    if not target and (not json_str):
        LOG.error('One of --target and --json must be provided.')
        cli_exit()
    if target:
        LOG.info('Create TF backend override')
        create_backend_override()
        LOG.info('Running `terraform init` with backend override')
        subprocess.check_call(['terraform', 'init', '-reconfigure', '-input=false', '-force-copy'])
        LOG.info("Running `terraform apply` on the target '%s'", target)
        subprocess.check_call(['terraform', 'apply', '-target', target, '-replace', target, '-auto-approve'])
        LOG.info('Generating terraform output')
        terraform_out = subprocess.check_output(['terraform', 'show', '-json'])
    if json_str:
        terraform_out = json_str
    LOG.info('Parsing terraform output')
    try:
        data_object = json.loads(terraform_out)
    except ValueError:
        LOG.error('Parsing JSON from terraform out unsuccessful!', exc_info=True)
        cli_exit()
    LOG.info('Find and copy built assets')
    find_and_copy_assets(directory_path, expression, data_object)