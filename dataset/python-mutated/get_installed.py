import os
import yaml
import six
from git.repo import Repo
from git.exc import InvalidGitRepositoryError
from st2common.runners.base_action import Action
from st2common.content.utils import get_packs_base_paths
from st2common.constants.pack import MANIFEST_FILE_NAME

class GetInstalled(Action):
    """Get information about installed pack."""

    def run(self, pack):
        if False:
            print('Hello World!')
        '\n        :param pack: Installed Pack Name to get info about\n        :type pack: ``str``\n        '
        packs_base_paths = get_packs_base_paths()
        pack_path = None
        metadata_file = None
        for packs_base_path in packs_base_paths:
            pack_path = os.path.join(packs_base_path, pack)
            pack_yaml_path = os.path.join(pack_path, MANIFEST_FILE_NAME)
            if os.path.isfile(pack_yaml_path):
                metadata_file = pack_yaml_path
                break
        if not os.path.isdir(pack_path):
            return {'pack': None, 'git_status': None}
        if not metadata_file:
            error = 'Pack "%s" doesn\'t contain pack.yaml file.' % pack
            raise Exception(error)
        try:
            details = self._parse_yaml_file(metadata_file)
        except Exception as e:
            error = 'Pack "%s" doesn\'t contain a valid pack.yaml file: %s' % (pack, six.text_type(e))
            raise Exception(error)
        try:
            repo = Repo(pack_path)
            git_status = 'Status:\n%s\n\nRemotes:\n%s' % (repo.git.status().split('\n')[0], '\n'.join([remote.url for remote in repo.remotes]))
            ahead_behind = repo.git.rev_list('--left-right', '--count', 'HEAD...origin/master').split()
            if ahead_behind != ['0', '0']:
                git_status += '\n\n'
                git_status += '%s commits ahead ' if ahead_behind[0] != '0' else ''
                git_status += 'and ' if '0' not in ahead_behind else ''
                git_status += '%s commits behind ' if ahead_behind[1] != '0' else ''
                git_status += 'origin/master.'
        except InvalidGitRepositoryError:
            git_status = None
        return {'pack': details, 'git_status': git_status}

    def _parse_yaml_file(self, file_path):
        if False:
            i = 10
            return i + 15
        with open(file_path) as data_file:
            details = yaml.safe_load(data_file)
        return details