import html
import json
import os
import re
import shlex
import subprocess
import markdown
from django.conf import settings
from zulip import Client
from zerver.models import get_realm
from zerver.openapi import markdown_extension
from zerver.openapi.curl_param_value_generators import AUTHENTICATION_LINE, assert_all_helper_functions_called
from zerver.openapi.openapi import get_endpoint_from_operationid

def test_generated_curl_examples_for_success(client: Client) -> None:
    if False:
        i = 10
        return i + 15
    default_authentication_line = f'{client.email}:{client.api_key}'
    realm = get_realm('zulip')
    md_engine = markdown.Markdown(extensions=[markdown_extension.makeExtension(api_url=realm.uri + '/api')])
    rest_endpoints_path = os.path.join(settings.DEPLOY_ROOT, 'api_docs/include/rest-endpoints.md')
    with open(rest_endpoints_path) as f:
        rest_endpoints_raw = f.read()
    ENDPOINT_REGEXP = re.compile('/api/\\s*(.*?)\\)')
    endpoint_list = sorted(set(re.findall(ENDPOINT_REGEXP, rest_endpoints_raw)))
    for endpoint in endpoint_list:
        article_name = endpoint + '.md'
        file_name = os.path.join(settings.DEPLOY_ROOT, 'api_docs/', article_name)
        if os.path.exists(file_name):
            with open(file_name) as f:
                curl_commands_to_test = [line for line in f if line.startswith('{generate_code_example(curl')]
        else:
            (endpoint_path, endpoint_method) = get_endpoint_from_operationid(endpoint)
            endpoint_string = endpoint_path + ':' + endpoint_method
            command = f'{{generate_code_example(curl)|{endpoint_string}|example}}'
            curl_commands_to_test = [command]
        for line in curl_commands_to_test:
            AUTHENTICATION_LINE[0] = default_authentication_line
            curl_command_html = md_engine.convert(line.strip())
            unescaped_html = html.unescape(curl_command_html)
            curl_regex = re.compile('<code>curl\\n(.*?)</code>', re.DOTALL)
            commands = re.findall(curl_regex, unescaped_html)
            for curl_command_text in commands:
                curl_command_text = curl_command_text.replace('BOT_EMAIL_ADDRESS:BOT_API_KEY', AUTHENTICATION_LINE[0])
                print('Testing {} ...'.format(curl_command_text.split('\n')[0]))
                generated_curl_command = [x for x in shlex.split(curl_command_text) if x != '\n']
                response_json = None
                response = None
                try:
                    response_json = subprocess.check_output(generated_curl_command, text=True)
                    response = json.loads(response_json)
                    assert response['result'] == 'success'
                except (AssertionError, Exception):
                    error_template = '\nError verifying the success of the API documentation curl example.\n\nFile: {file_name}\nLine: {line}\nCurl command:\n{curl_command}\nResponse:\n{response}\n\nThis test is designed to check each generate_code_example(curl) instance in the\nAPI documentation for success. If this fails then it means that the curl example\nthat was generated was faulty and when tried, it resulted in an unsuccessful\nresponse.\n\nCommon reasons for why this could occur:\n    1. One or more example values in zerver/openapi/zulip.yaml for this endpoint\n       do not line up with the values in the test database.\n    2. One or more mandatory parameters were included in the "exclude" list.\n\nTo learn more about the test itself, see zerver/openapi/test_curl_examples.py.\n'
                    print(error_template.format(file_name=file_name, line=line, curl_command=generated_curl_command, response=response_json if response is None else json.dumps(response, indent=4)))
                    raise
    assert_all_helper_functions_called()