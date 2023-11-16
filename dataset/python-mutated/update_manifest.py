import os
import subprocess
import sys
from typing import Optional
import yaml

def latest_commit_id() -> str:
    if False:
        for i in range(10):
            print('nop')
    cmd = 'git log --format="%H" -n 1'
    commit_id = subprocess.check_output(cmd, shell=True)
    return commit_id.decode('utf-8').strip()

def update_manifest(docker_tag: Optional[str]) -> None:
    if False:
        while True:
            i = 10
    'Update manifest_template file with latest commit hash.'
    commit_id = latest_commit_id()
    template_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../'))
    template_filepath = os.path.join(template_dir, 'hagrid/manifest_template.yml')
    with open(template_filepath) as stream:
        template_dict = yaml.safe_load(stream)
    template_dict['hash'] = commit_id
    if docker_tag:
        template_dict['dockerTag'] = docker_tag
    with open(template_filepath, 'w') as fp:
        yaml.dump(template_dict, fp, sort_keys=False)
if __name__ == '__main__':
    docker_tag = None
    if len(sys.argv) > 1:
        docker_tag = sys.argv[1]
    update_manifest(docker_tag)