import os
import re
import requests
import tempfile
from urllib.parse import urlparse
import textract
from ansible.module_utils.basic import AnsibleModule
from lwe.core.config import Config
from lwe.core.logger import Logger
config = Config()
config.set('debug.log.enabled', True)
log = Logger('text_extractor', config)
SUPPORTED_FILE_EXTENSIONS = ['.docx', '.pptx', '.xlsx', '.odt', '.ods', '.odp', '.odg', '.odc', '.odf', '.odi', '.odm', '.pdf', '.rtf', '.md', '.epub', '.txt', '.csv', '.html', '.htm', '.xhtml', '.xml', '.eml', '.msg']
DOCUMENTATION = '\n---\nmodule: text_extractor\nshort_description: Extract text content from a file or URL\ndescription:\n    - This module extracts the main text content from a given file or URL\n    - For URLs, it extracts the main text content from the page, excluding header and footer.\n    - For files, see SUPPORTED_FILE_EXTENSIONS in the module code.\noptions:\n    path:\n      description:\n          - The path to the file or the URL of the HTML content.\n      type: path\n      required: true\n    max_length:\n      description:\n          - Limit the return of the extracted content to this length.\n      type: int\n      required: false\nauthor:\n    - Chad Phillips (@thehunmonkgroup)\n'
EXAMPLES = '\n  - name: Extract content from a local PDF file\n    text_extractor:\n      path: "/path/to/your/file.pdf"\n\n  - name: Extract content from a URL\n    text_extractor:\n      path: "https://example.com/sample.html"\n      max_length: 3000\n'
RETURN = '\n  content:\n      description: The extracted main text content from the HTML.\n      type: str\n      returned: success\n  length:\n      description: The length of the extracted main text content.\n      type: int\n      returned: success\n'

def extract_text(path):
    if False:
        while True:
            i = 10
    text = textract.process(path).decode('utf-8')
    return text

def main():
    if False:
        print('Hello World!')
    result = dict(changed=False, response=dict())
    module = AnsibleModule(argument_spec=dict(path=dict(type='path', required=True), max_length=dict(type='int', required=False)), supports_check_mode=True)
    path = module.params['path']
    max_length = module.params['max_length']
    if module.check_mode:
        module.exit_json(**result)
    parsed_url = urlparse(path)
    is_url = parsed_url.scheme in ['http', 'https']
    cleanup_tmpfile_path = None
    if is_url:
        try:
            response = requests.get(path)
            response.raise_for_status()
            content = response.text
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
                f.write(content)
                path = cleanup_tmpfile_path = f.name
        except Exception as e:
            message = f'Error downloading content from URL {path}: {str(e)}'
            log.error(message)
            module.fail_json(msg=message)
    if not os.access(path, os.R_OK):
        message = f'File not found or not readable: {path}'
        log.error(message)
        module.fail_json(msg=message)
    (_, file_extension) = os.path.splitext(path)
    file_extension = file_extension.lower()
    if file_extension in SUPPORTED_FILE_EXTENSIONS:
        try:
            content = extract_text(path)
        except Exception as e:
            message = f'Error extracting {file_extension} content: {str(e)}'
            log.error(message)
            module.fail_json(msg=message)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = re.sub('[^\\x00-\\x7F]+', '', f.read())
    if max_length:
        content = content[:max_length]
    if cleanup_tmpfile_path:
        os.remove(cleanup_tmpfile_path)
    result['content'] = content
    result['length'] = len(content)
    log.info('Content extracted successfully')
    module.exit_json(**result)
if __name__ == '__main__':
    main()