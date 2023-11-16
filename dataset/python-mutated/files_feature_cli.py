from hatch.template import File
from hatch.utils.fs import Path

class PackageEntryPoint(File):
    TEMPLATE = 'import sys\n\nif __name__ == "__main__":\n    from {package_name}.cli import {package_name}\n\n    sys.exit({package_name}())\n'

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            return 10
        super().__init__(Path(template_config['package_name'], '__main__.py'), self.TEMPLATE.format(**template_config))

class CommandLinePackage(File):
    TEMPLATE = 'import click\n\nfrom {package_name}.__about__ import __version__\n\n\n@click.group(context_settings={{"help_option_names": ["-h", "--help"]}}, invoke_without_command=True)\n@click.version_option(version=__version__, prog_name="{project_name}")\ndef {package_name}():\n    click.echo("Hello world!")\n'

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            i = 10
            return i + 15
        super().__init__(Path(template_config['package_name'], 'cli', '__init__.py'), self.TEMPLATE.format(**template_config))