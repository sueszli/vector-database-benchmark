import glob
import os
import pathlib
import re
import signal
import subprocess
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from time import sleep
from typing import Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from dagster_ui_screenshot.defaults import DEFAULT_OUTPUT_ROOT
DAGIT_ROUTE_LOAD_TIME = 2
DAGIT_STARTUP_TIME = 6
DOWNLOAD_SVG_TIME = 2
SVG_ROOT = os.path.join(DEFAULT_OUTPUT_ROOT, 'asset-screenshots')
CODE_SAMPLES_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'examples', 'docs_snippets', 'docs_snippets')
SVG_FONT_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'static', 'font_info.svg')
with open(SVG_FONT_DATA_FILE, 'r', encoding='utf-8') as f:
    SVG_FONT_DATA = f.read()

def _add_font_info_to_svg(svg_filepath: str):
    if False:
        return 10
    'Adds embedded Dagster font information to an SVG file downloaded from Dagit.'
    with open(svg_filepath, 'r', encoding='utf-8') as f:
        svg = f.read()
    with open(svg_filepath, 'w', encoding='utf-8') as f:
        f.write(svg.replace('<style xmlns="http://www.w3.org/1999/xhtml"></style>', SVG_FONT_DATA))

def _get_latest_download(file_extension: str) -> str:
    if False:
        return 10
    'Returns the path to the most recently downloaded file with the given extension.'
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    list_of_downloads = glob.glob(downloads_folder + f'/*.{file_extension}')
    return max(list_of_downloads, key=os.path.getctime)

@contextmanager
def _setup_snippet_file(code_path: str, snippet_fn: Optional[str]):
    if False:
        for i in range(10):
            print('nop')
    'Creates a temporary file that contains the contents of the given code file,\n    setting up the given snippet function as a repository if specified.\n    '
    with TemporaryDirectory() as temp_dir:
        with open(code_path, 'r', encoding='utf-8') as f:
            code = f.read()
        if snippet_fn:
            code = f'{code}\n\nfrom dagster import repository\n@repository\ndef demo_repo():\n    return {snippet_fn}()\n'
        temp_code_file = os.path.join(temp_dir, 'code.py')
        with open(temp_code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        yield temp_code_file

def generate_svg_for_file(code_path: str, destination_path: str, snippet_fn: Optional[str]):
    if False:
        while True:
            i = 10
    'Generates an SVG for the given code file & entry function, saving it to the given destination path.'
    driver = None
    dagit_process = None
    try:
        with _setup_snippet_file(code_path, snippet_fn) as temp_code_file:
            command = ['dagit', '-f', temp_code_file]
            dagit_process = subprocess.Popen(command)
            sleep(DAGIT_STARTUP_TIME)
            driver = webdriver.Chrome()
            driver.set_window_size(1024, 768)
            driver.get('http://localhost:3000')
            driver.execute_script("window.localStorage.setItem('communityNux','1')")
            driver.refresh()
            sleep(DAGIT_ROUTE_LOAD_TIME)
            element = driver.find_element(By.XPATH, '//div[@aria-label="download_for_offline"]')
            element.click()
            sleep(DOWNLOAD_SVG_TIME)
            downloaded_file = _get_latest_download('svg')
            pathlib.Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
            output_file = destination_path
            os.rename(downloaded_file, output_file)
            _add_font_info_to_svg(output_file)
    finally:
        if driver:
            driver.quit()
        if dagit_process:
            dagit_process.send_signal(signal.SIGINT)
            dagit_process.wait()

def parse_params(param_str: str) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    'Parses a set of params for a markdown code block.\n\n    For example, returns {"foo": "bar", "baz": "qux"} for:\n\n    ```python\n    foo=bar baz=qux.\n    ```\n    '
    params = re.split('\\s+', param_str)
    return {param.split('=')[0]: param.split('=')[1] for param in params if len(param) > 0}

def generate_svg(target_mdx_file: str):
    if False:
        return 10
    with open(target_mdx_file, 'r', encoding='utf-8') as f:
        snippets = [parse_params(x) for x in re.findall('```python([^\\n]+dagimage[^\\n]+)', f.read())]
    updated_snippet_params = []
    for snippet_params in snippets:
        filepath = snippet_params['file']
        snippet_fn = snippet_params.get('function')
        destination_file_path = f".{filepath[:-3]}{('/' + snippet_fn if snippet_fn else '')}.svg"
        generate_svg_for_file(os.path.join(CODE_SAMPLES_ROOT, f'.{filepath}'), os.path.join(SVG_ROOT, destination_file_path), snippet_fn)
        updated_snippet_params.append({**snippet_params, 'dagimage': os.path.normpath(os.path.join('images', 'asset-screenshots', destination_file_path))})
    with open(target_mdx_file, 'r', encoding='utf-8') as f:
        pattern = re.compile('(```python)([^\\n]+dagimage[^\\n]+)', re.S)
        idx = [0]

        def _replace(match):
            if False:
                print('Hello World!')
            snippet_parmas = updated_snippet_params[idx[0]]
            snippet_params_text = ' '.join((f'{k}={v}' for (k, v) in snippet_parmas.items()))
            out = f'{match.group(1)} {snippet_params_text}'
            idx[0] += 1
            return out
        updated_mdx_contents = re.sub(pattern, _replace, f.read())
    with open(target_mdx_file, 'w', encoding='utf-8') as f:
        f.write(updated_mdx_contents)