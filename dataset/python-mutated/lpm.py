import itertools
import logging
from multiprocessing.pool import ThreadPool
from typing import List, Optional
import click
from rich.console import Console
from localstack import config
from localstack.cli.exceptions import CLIError
from localstack.packages import InstallTarget, Package
from localstack.packages.api import NoSuchPackageException, PackagesPluginManager
from localstack.utils.bootstrap import setup_logging
LOG = logging.getLogger(__name__)
console = Console()

@click.group()
def cli():
    if False:
        print('Hello World!')
    '\n    The LocalStack Package Manager (lpm) CLI is a set of commands to install third-party packages used by localstack\n    service providers.\n\n    Here are some handy commands:\n\n    List all packages\n\n        python -m localstack.cli.lpm list\n\n    Install DynamoDB Local:\n\n        python -m localstack.cli.install dynamodb-local\n\n    Install all community packages, four in parallel:\n\n        python -m localstack.cli.lpm list | grep "/community" | cut -d\'/\' -f1 | xargs python -m localstack.cli.lpm install --parallel 4\n    '
    setup_logging()

def _do_install_package(package: Package, version: str=None, target: InstallTarget=None):
    if False:
        return 10
    console.print(f'installing... [bold]{package}[/bold]')
    try:
        package.install(version=version, target=target)
        console.print(f'[green]installed[/green] [bold]{package}[/bold]')
    except Exception as e:
        console.print(f'[red]error[/red] installing {package}: {e}')
        raise e

@cli.command()
@click.argument('package', nargs=-1, required=True)
@click.option('--parallel', type=int, default=1, required=False, help='how many installers to run in parallel processes')
@click.option('--version', type=str, default=None, required=False, help='version to install of a package')
@click.option('--target', type=click.Choice([target.name.lower() for target in InstallTarget]), default=None, required=False, help='target of the installation')
def install(package: List[str], parallel: Optional[int]=1, version: Optional[str]=None, target: Optional[str]=None):
    if False:
        while True:
            i = 10
    'Install one or more packages.'
    try:
        if target:
            target = InstallTarget[str.upper(target)]
        else:
            target = InstallTarget.STATIC_LIBS
        console.print(f'resolving packages: {package}')
        package_manager = PackagesPluginManager()
        package_manager.load_all()
        package_instances = package_manager.get_packages(package, version)
        if parallel > 1:
            console.print(f'install {parallel} packages in parallel:')
        config.dirs.mkdirs()
        with ThreadPool(processes=parallel) as pool:
            pool.starmap(_do_install_package, zip(package_instances, itertools.repeat(version), itertools.repeat(target)))
    except NoSuchPackageException as e:
        LOG.debug(str(e), exc_info=e)
        raise CLIError(str(e))
    except Exception as e:
        LOG.debug('one or more package installations failed.', exc_info=e)
        raise CLIError('one or more package installations failed.')

@cli.command(name='list')
@click.option('-v', '--verbose', is_flag=True, default=False, required=False, help='Verbose output (show additional info on packages)')
def list_packages(verbose: bool):
    if False:
        print('Hello World!')
    'List available packages of all repositories'
    package_manager = PackagesPluginManager()
    package_manager.load_all()
    packages = package_manager.get_all_packages()
    for (package_name, package_scope, package_instance) in packages:
        console.print(f'[green]{package_name}[/green]/{package_scope}')
        if verbose:
            for version in package_instance.get_versions():
                if version == package_instance.default_version:
                    console.print(f'  - [bold]{version} (default)[/bold]', highlight=False)
                else:
                    console.print(f'  - {version}', highlight=False)
if __name__ == '__main__':
    cli()