from __future__ import annotations
import itertools
import subprocess
import rich_click as click
PYTHON_VERSIONS = ['3.8', '3.9', '3.10', '3.11']
GHCR_IO_PREFIX = 'ghcr.io'
GHCR_IO_IMAGES = ['{prefix}/{repo}/{branch}/ci/python{python}:latest', '{prefix}/{repo}/{branch}/prod/python{python}:latest']

def pull_push_all_images(source_prefix: str, target_prefix: str, images: list[str], source_branch: str, source_repo: str, target_branch: str, target_repo: str):
    if False:
        while True:
            i = 10
    for (python, image) in itertools.product(PYTHON_VERSIONS, images):
        source_image = image.format(prefix=source_prefix, branch=source_branch, repo=source_repo, python=python)
        target_image = image.format(prefix=target_prefix, branch=target_branch, repo=target_repo, python=python)
        print(f'Copying image: {source_image} -> {target_image}')
        subprocess.run(['regctl', 'image', 'copy', '--force-recursive', '--digest-tags', source_image, target_image], check=True)

@click.group(invoke_without_command=True)
@click.option('--source-branch', type=str, default='main', help='Source branch name [main]')
@click.option('--target-branch', type=str, default='main', help='Target branch name [main]')
@click.option('--source-repo', type=str, default='apache/airflow', help='Source repo')
@click.option('--target-repo', type=str, default='apache/airflow', help='Target repo')
def main(source_branch: str, target_branch: str, source_repo: str, target_repo: str):
    if False:
        for i in range(10):
            print('nop')
    pull_push_all_images(GHCR_IO_PREFIX, GHCR_IO_PREFIX, GHCR_IO_IMAGES, source_branch, source_repo, target_branch, target_repo)
if __name__ == '__main__':
    main()