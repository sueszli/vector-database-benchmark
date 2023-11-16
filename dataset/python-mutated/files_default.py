from hatch.template import File
from hatch.utils.fs import Path

class PackageRoot(File):

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            i = 10
            return i + 15
        super().__init__(Path(template_config['package_name'], '__init__.py'), '')

class MetadataFile(File):

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            while True:
                i = 10
        super().__init__(Path(template_config['package_name'], '__about__.py'), '__version__ = "0.0.1"\n')

class Readme(File):
    TEMPLATE = '# {project_name}\n\n[![PyPI - Version](https://img.shields.io/pypi/v/{project_name_normalized}.svg)](https://pypi.org/project/{project_name_normalized})\n[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/{project_name_normalized}.svg)](https://pypi.org/project/{project_name_normalized})\n{extra_badges}\n-----\n\n**Table of Contents**\n\n- [Installation](#installation)\n{extra_toc}\n## Installation\n\n```console\npip install {project_name_normalized}\n```{license_info}\n'

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            print('Hello World!')
        extra_badges = ''
        extra_toc = ''
        license_info = ''
        if template_config['license_data']:
            extra_toc += '- [License](#license)\n'
            license_info += f"\n\n## License\n\n`{template_config['project_name_normalized']}` is distributed under the terms of "
            license_data = template_config['license_data']
            if len(license_data) == 1:
                license_id = next(iter(license_data))
                license_info += f'the [{license_id}](https://spdx.org/licenses/{license_id}.html) license.'
            else:
                license_info += 'any of the following licenses:\n'
                for license_id in sorted(license_data):
                    license_info += f'\n- [{license_id}](https://spdx.org/licenses/{license_id}.html)'
        super().__init__(Path(template_config['readme_file_path']), self.TEMPLATE.format(extra_badges=extra_badges, extra_toc=extra_toc, license_info=license_info, **template_config))

class PyProject(File):
    TEMPLATE = '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n[project]\nname = "{project_name_normalized}"\ndynamic = ["version"]\ndescription = {description!r}\nreadme = "{readme_file_path}"\nrequires-python = ">=3.8"\nlicense = "{license_expression}"{license_files}\nkeywords = []\nauthors = [\n  {{ name = "{name}", email = "{email}" }},\n]\nclassifiers = [\n  "Development Status :: 4 - Beta",\n  "Programming Language :: Python",\n  "Programming Language :: Python :: 3.8",\n  "Programming Language :: Python :: 3.9",\n  "Programming Language :: Python :: 3.10",\n  "Programming Language :: Python :: 3.11",\n  "Programming Language :: Python :: 3.12",\n  "Programming Language :: Python :: Implementation :: CPython",\n  "Programming Language :: Python :: Implementation :: PyPy",\n]\ndependencies = {dependency_data}\n\n[project.urls]{project_url_data}{cli_scripts}\n\n[tool.hatch.version]\npath = "{package_metadata_file_path}"{tests_section}\n'

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            print('Hello World!')
        template_config = dict(template_config)
        template_config['name'] = repr(template_config['name'])[1:-1]
        project_url_data = ''
        project_urls = plugin_config['project_urls'] if 'project_urls' in plugin_config else {'Documentation': 'https://github.com/unknown/{project_name_normalized}#readme', 'Issues': 'https://github.com/unknown/{project_name_normalized}/issues', 'Source': 'https://github.com/unknown/{project_name_normalized}'}
        if project_urls:
            for (label, url) in project_urls.items():
                normalized_label = f'"{label}"' if ' ' in label else label
                project_url_data += f'\n{normalized_label} = "{url.format(**template_config)}"'
        dependency_data = '['
        if template_config['dependencies']:
            for dependency in sorted(template_config['dependencies']):
                dependency_data += f'\n  "{dependency}",\n'
        dependency_data += ']'
        cli_scripts = ''
        if template_config['args']['cli']:
            cli_scripts = f'''\n\n[project.scripts]\n{template_config['project_name_normalized']} = "{template_config['package_name']}.cli:{template_config['package_name']}"'''
        tests_section = ''
        if plugin_config['tests']:
            package_location = 'src/' if plugin_config['src-layout'] else ''
            tests_section = f'''\n\n[tool.hatch.envs.default]\ndependencies = [\n  "coverage[toml]>=6.5",\n  "pytest",\n]\n[tool.hatch.envs.default.scripts]\ntest = "pytest {{args:tests}}"\ntest-cov = "coverage run -m pytest {{args:tests}}"\ncov-report = [\n  "- coverage combine",\n  "coverage report",\n]\ncov = [\n  "test-cov",\n  "cov-report",\n]\n\n[[tool.hatch.envs.all.matrix]]\npython = ["3.8", "3.9", "3.10", "3.11", "3.12"]\n\n[tool.hatch.envs.lint]\ndetached = true\ndependencies = [\n  "black>=23.1.0",\n  "mypy>=1.0.0",\n  "ruff>=0.0.243",\n]\n[tool.hatch.envs.lint.scripts]\ntyping = "mypy --install-types --non-interactive {{args:{package_location}{template_config['package_name']} tests}}"\nstyle = [\n  "ruff {{args:.}}",\n  "black --check --diff {{args:.}}",\n]\nfmt = [\n  "black {{args:.}}",\n  "ruff --fix {{args:.}}",\n  "style",\n]\nall = [\n  "style",\n  "typing",\n]\n\n[tool.black]\nline-length = 120\nskip-string-normalization = true\n\n[tool.ruff]\nline-length = 120\nselect = [\n  "A",\n  "ARG",\n  "B",\n  "C",\n  "DTZ",\n  "E",\n  "EM",\n  "F",\n  "FBT",\n  "I",\n  "ICN",\n  "ISC",\n  "N",\n  "PLC",\n  "PLE",\n  "PLR",\n  "PLW",\n  "Q",\n  "RUF",\n  "S",\n  "T",\n  "TID",\n  "UP",\n  "W",\n  "YTT",\n]\nignore = [\n  # Allow non-abstract empty methods in abstract base classes\n  "B027",\n  # Allow boolean positional values in function calls, like `dict.get(... True)`\n  "FBT003",\n  # Ignore checks for possible passwords\n  "S105", "S106", "S107",\n  # Ignore complexity\n  "C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915",\n]\nunfixable = [\n  # Don't touch unused imports\n  "F401",\n]\n\n[tool.ruff.isort]\nknown-first-party = ["{template_config['package_name']}"]\n\n[tool.ruff.flake8-tidy-imports]\nban-relative-imports = "all"\n\n[tool.ruff.per-file-ignores]\n# Tests can use magic values, assertions, and relative imports\n"tests/**/*" = ["PLR2004", "S101", "TID252"]\n\n[tool.coverage.run]\nsource_pkgs = ["{template_config['package_name']}", "tests"]\nbranch = true\nparallel = true\nomit = [\n  "{package_location}{template_config['package_name']}/__about__.py",\n]\n\n[tool.coverage.paths]\n{template_config['package_name']} = ["{package_location}{template_config['package_name']}", "*/{template_config['project_name_normalized']}/{package_location}{template_config['package_name']}"]\ntests = ["tests", "*/{template_config['project_name_normalized']}/tests"]\n\n[tool.coverage.report]\nexclude_lines = [\n  "no cov",\n  "if __name__ == .__main__.:",\n  "if TYPE_CHECKING:",\n]'''
        super().__init__(Path('pyproject.toml'), self.TEMPLATE.format(project_url_data=project_url_data, dependency_data=dependency_data, cli_scripts=cli_scripts, tests_section=tests_section, **template_config))