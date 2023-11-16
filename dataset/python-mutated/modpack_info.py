"""
Modpack definition file.
"""
import toml
from ..data_definition import DataDefinition
FILE_VERSION = '2'

class ModpackInfo(DataDefinition):
    """
    Represents the header file of the modpack. Contains info for loading data
    and about the creators of the modpack.
    """

    def __init__(self, targetdir: str, filename: str):
        if False:
            print('Hello World!')
        super().__init__(targetdir, filename)
        self.packagename: str = None
        self.version: str = None
        self.versionstr: str = None
        self.extra_info: dict[str, str] = {}
        self.includes: list[str] = []
        self.excludes: list[str] = []
        self.requires: list[str] = []
        self.conflicts: list[str] = []
        self.authors: dict[str, str] = {}
        self.author_groups: dict[str, str] = {}

    def add_author(self, name: str, fullname: str=None, since: str=None, until: str=None, roles: str=None, contact: str=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds an author with optional contact info.\n\n        :param name: Nickname of the author. Must be unique for the modpack.\n        :type name: str\n        :param fullname: Full name of the author.\n        :type fullname: str\n        :param since: Version number of the release where the author started to contribute.\n        :type since: str\n        :param until: Version number of the release where the author stopped to contribute.\n        :type until: str\n        :param roles: List of roles of the author during the creation of the modpack.\n        :type roles: list\n        :param contact: Dictionary with contact info. See the spec\n                        for available parameters.\n        :type contact: dict\n        '
        author = {}
        author['name'] = name
        if fullname:
            author['fullname'] = fullname
        if since:
            author['since'] = since
        if until:
            author['until'] = until
        if roles:
            author['roles'] = roles
        if contact:
            author['contact'] = contact
        self.authors[name] = author

    def add_author_group(self, name: str, authors: list[str], description: str=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds an author with optional contact info.\n\n        :param name: Group or team name.\n        :type name: str\n        :param authors: List of author identifiers. These must match up\n                        with subtable keys in the self.authors.\n        :type authors: list\n        :param description: Path to a file with a description of the team.\n        :type description: str\n        '
        author_group = {}
        author_group['name'] = name
        author_group['authors'] = authors
        if description:
            author_group['description'] = description
        self.author_groups[name] = author_group

    def add_include(self, path: str) -> None:
        if False:
            return 10
        '\n        Add a path to an asset that is loaded by the modpack.\n\n        :param path: Path to assets that should be mounted on load time.\n        :type path: str\n        '
        self.includes.append(path)

    def add_exclude(self, path: str) -> None:
        if False:
            return 10
        '\n        Add a path to an asset that excluded from loading.\n\n        :param path: Path to assets.\n        :type path: str\n        '
        self.excludes.append(path)

    def add_conflict(self, modpack_id: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add an identifier of another modpack that has a conflict with this modpack.\n\n        :param modpack_id: Modpack alias or identifier.\n        :type modpack_id: str\n        '
        self.conflicts.append(modpack_id)

    def add_dependency(self, modpack_id: str) -> None:
        if False:
            print('Hello World!')
        '\n        Add an identifier of another modpack that is a dependency of this modpack.\n\n        :param modpack_id: Modpack alias or identifier.\n        :type modpack_id: str\n        '
        self.requires.append(modpack_id)

    def set_info(self, packagename: str, version: str, versionstr: str=None, repo: str=None, alias: str=None, title: str=None, description: str=None, long_description: str=None, url: str=None, licenses: str=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Set the general information about the modpack.\n\n        :param packagename: Name of the modpack.\n        :type packagename: str\n        :param version: Internal version number. Must have semver format.\n        :type version: str\n        :param versionstr: Human-readable version number.\n        :type versionstr: str\n        :param repo: Name of the repo where the package is hosted.\n        :type repo: str\n        :param alias: Alias of the modpack.\n        :type alias: str\n        :param title: Title used in UI.\n        :type title: str\n        :param description: Path to a file with a short description (max 500 chars).\n        :type description: str\n        :param long_description: Path to a file with a detailed description.\n        :type long_description: str\n        :param url: Link to the modpack's website.\n        :type url: str\n        :param licenses: License(s) of the modpack.\n        :type licenses: list\n        "
        self.packagename = packagename
        self.version = version
        if versionstr:
            self.extra_info['versionstr'] = versionstr
        if repo:
            self.extra_info['repo'] = repo
        if alias:
            self.extra_info['alias'] = alias
        if title:
            self.extra_info['title'] = title
        if description:
            self.extra_info['description'] = description
        if long_description:
            self.extra_info['long_description'] = long_description
        if url:
            self.extra_info['url'] = url
        if licenses:
            self.extra_info['licenses'] = licenses

    def dump(self) -> str:
        if False:
            return 10
        '\n        Outputs the modpack info to the TOML output format.\n        '
        output_str = '# openage modpack definition file\n\n'
        output_dict = {}
        output_dict.update({'file_version': FILE_VERSION})
        if not self.packagename:
            raise RuntimeError(f'{self}: packagename needs to be defined before dumping.')
        if not self.version:
            raise RuntimeError(f'{self}: version needs to be defined before dumping.')
        info_table = {'info': {}}
        info_table['info'].update({'name': self.packagename, 'version': self.version})
        info_table['info'].update(self.extra_info)
        output_dict.update(info_table)
        assets_table = {'assets': {}}
        assets_table['assets'].update({'include': self.includes, 'exclude': self.excludes})
        output_dict.update(assets_table)
        dependency_table = {'dependency': {}}
        dependency_table['dependency'].update({'modpacks': self.requires})
        output_dict.update(dependency_table)
        conflicts_table = {'conflict': {}}
        conflicts_table['conflict'].update({'modpacks': self.conflicts})
        output_dict.update(conflicts_table)
        authors_table = {'authors': {}}
        authors_table['authors'].update(self.authors)
        output_dict.update(authors_table)
        authorgroups_table = {'authorgroups': {}}
        authorgroups_table['authorgroups'].update(self.author_groups)
        output_dict.update(authorgroups_table)
        output_str += toml.dumps(output_dict)
        return output_str

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'ModpackInfo<{self.packagename}>'