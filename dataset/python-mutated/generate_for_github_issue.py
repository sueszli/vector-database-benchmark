from collections import OrderedDict
import os
import git
import requests
import subprocess
import markdown_to_json
import config.logging_config as logging_config
from code_gen.generate import main
logger = logging_config.getLogger(__name__)

def run_command(command):
    if False:
        return 10
    subprocess.run(command, shell=True)

def extract_prompts(markdown_dict):
    if False:
        i = 10
        return i + 15
    prompts = []
    for (key, value) in markdown_dict.items():
        if key == 'prompt':
            prompts.append(value)
        elif isinstance(value, (dict, OrderedDict)):
            prompts.extend(extract_prompts(value))
    return prompts

def get_all_docs_urls(markdown_dict):
    if False:
        while True:
            i = 10

    def extract_urls(markdown_dict):
        if False:
            for i in range(10):
                print('nop')
        urls = []
        for (key, value) in markdown_dict.items():
            if key == 'urls':
                urls.extend(value)
            elif isinstance(value, (dict, OrderedDict)):
                urls.extend(extract_urls(value))
        return urls
    all_urls = extract_urls(markdown_dict)
    return unique_urls(all_urls)

def unique_urls(urls):
    if False:
        while True:
            i = 10
    return list(set([url.split('#')[0] for url in urls]))

def generate_app_file_prompt(requirements, app_file_content):
    if False:
        while True:
            i = 10
    if app_file_content:
        return f'Given the existing app file and the requirements below, generate an app file that provides propDefinitions and methods that solve the requirements:\n## EXISTING APP FILE CODE\n\n{requirements}\n\n## REQUIREMENTS\n\n{app_file_content}'
    return f'Generate an app file that provides propDefinitions and methods that solve the following requirements:\n\n{requirements}'

def generate(issue_number, output_dir, generate_pr=True, clean=False, verbose=False, tries=3, remote_name='origin'):
    if False:
        for i in range(10):
            print('nop')
    repo_path = os.path.abspath(os.path.join('..', '..'))
    output_dir = os.path.abspath(output_dir)
    if generate_pr:
        output_dir = os.path.join(repo_path, 'components')
        repo = git.Repo(repo_path)
        if not clean and repo.index.diff(None):
            logger.warn('Your git stage is not clean. Please stash/commit your changes or use --clean to discard them')
            return
        branch_name = f'issue-{issue_number}'
        run_command(f'git fetch {remote_name}')
        if any((reference.name == branch_name for reference in repo.references)):
            run_command(f'git checkout {branch_name}')
        else:
            run_command(f'git checkout -b {branch_name}')
        run_command(f'git reset --hard {remote_name}/master')
    md = requests.get(f'https://api.github.com/repos/PipedreamHQ/pipedream/issues/{issue_number}').json()['body'].lower()
    markdown = markdown_to_json.dictify(md)
    app = list(markdown.keys())[0]
    global_urls = []
    requirements = []
    app_base_path = os.path.join(output_dir, app)
    file_path = os.path.join(app_base_path, f'{app}.app.mjs')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    app_file_content = None
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            logger.debug('Reading existing app file')
            app_file_content = f.read()
    else:
        logger.debug('No existing app file found, creating new one')
    app_file_instructions = generate_app_file_prompt('\n\n'.join(extract_prompts(markdown)), app_file_content)
    all_docs_urls = get_all_docs_urls(markdown)
    logger.debug('Generating app file')
    app_file_content = main('app', app, instructions=app_file_instructions, tries=tries, urls=all_docs_urls, verbose=verbose)
    with open(file_path, 'w') as f:
        logger.debug('Writing app file')
        f.write(app_file_content)
    for h2_header in markdown[app]:
        if h2_header == 'urls':
            global_urls += markdown[app][h2_header]
            continue
        for component_key in markdown[app][h2_header]:
            component_data = markdown[app][h2_header][component_key]
            instructions = f"### Requirements\n\n{component_data['prompt']}\n\n### Use methods and propDefinitions from this app file\n\nUse the methods and propDefinitions in this app file to solve the requirements:\n\n{app_file_content}\n\nYou can call methods from the app file using `this.{app}.<method name>`. Think about it: you've already defined props and methods in the app file, so you should use these to promote code reuse.\n\n"
            urls = component_data.get('urls')
            if not urls:
                urls = []
                logger.warn(f'No API docs URLs found for {component_key}')
            if 'source' in h2_header:
                component_type = 'webhook_source' if 'webhook' in h2_header else 'polling_source'
            elif 'action' in h2_header:
                component_type = 'action'
            else:
                continue
            requirements.append({'type': component_type, 'key': component_key, 'instructions': f'The component key is {app}-{component_key}. {instructions}', 'urls': unique_urls(global_urls + urls)})
    for component in requirements:
        logger.info(f"generating {component['key']}...")
        result = main(component['type'], app, component['instructions'], tries=tries, urls=component['urls'], verbose=verbose)
        component_type = 'sources' if 'source' in component['type'] else 'actions'
        file_path = f"{output_dir}/{app}/{component_type}/{component['key']}/{component['key']}.mjs"
        logger.info(f'writing output to {file_path}')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(result)
    if generate_pr:
        run_command(f'npx eslint {app_base_path} --fix')
        run_command(f'git add -f {app_base_path}')
        run_command(f"git commit --no-verify -m '{app} init'")
        run_command(f'git push -f --no-verify --set-upstream {remote_name} {branch_name}')
        run_command(f"gh pr create -d -l ai-assisted -t 'New Components - {app}' -b 'Resolves #{issue_number}.'")