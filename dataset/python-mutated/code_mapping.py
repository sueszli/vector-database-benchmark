from __future__ import annotations
import logging
from typing import Dict, List, NamedTuple, Tuple, Union
from sentry.models.integrations.organization_integration import OrganizationIntegration
from sentry.models.integrations.repository_project_path_config import RepositoryProjectPathConfig
from sentry.models.project import Project
from sentry.models.repository import Repository
from sentry.services.hybrid_cloud.integration.model import RpcOrganizationIntegration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
EXTENSIONS = ['js', 'jsx', 'tsx', 'ts', 'mjs', 'py', 'rb', 'rake', 'php', 'go']
FILE_PATH_PREFIX_LENGTH = {'app:///': 7, '../': 3, './': 2}
MAX_CONNECTION_ERRORS = 10

class Repo(NamedTuple):
    name: str
    branch: str

class RepoTree(NamedTuple):
    repo: Repo
    files: List[str]

class CodeMapping(NamedTuple):
    repo: Repo
    stacktrace_root: str
    source_path: str

class UnsupportedFrameFilename(Exception):
    pass

def get_extension(file_path: str) -> str:
    if False:
        print('Hello World!')
    extension = ''
    if file_path:
        ext_period = file_path.rfind('.')
        if ext_period >= 1:
            extension = file_path.rsplit('.')[-1]
    return extension

def should_include(file_path: str) -> bool:
    if False:
        print('Hello World!')
    include = True
    if file_path.endswith('spec.jsx') or file_path.startswith('tests/'):
        include = False
    return include

def get_straight_path_prefix_end_index(file_path: str) -> int:
    if False:
        print('Hello World!')
    index = 0
    for prefix in FILE_PATH_PREFIX_LENGTH:
        while file_path.startswith(prefix):
            index += FILE_PATH_PREFIX_LENGTH[prefix]
            file_path = file_path[FILE_PATH_PREFIX_LENGTH[prefix]:]
    return index

def remove_straight_path_prefix(file_path: str) -> str:
    if False:
        while True:
            i = 10
    return file_path[get_straight_path_prefix_end_index(file_path):]

def filter_source_code_files(files: List[str]) -> List[str]:
    if False:
        print('Hello World!')
    '\n    This takes the list of files of a repo and returns\n    the file paths for supported source code files\n    '
    _supported_files = []
    for file_path in files:
        try:
            extension = get_extension(file_path)
            if extension in EXTENSIONS and should_include(file_path):
                _supported_files.append(file_path)
        except Exception:
            logger.exception("We've failed to store the file path.")
    return _supported_files

class FrameFilename:

    def __init__(self, frame_file_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if frame_file_path[0] == '/':
            frame_file_path = frame_file_path.replace('/', '', 1)
        if not frame_file_path or frame_file_path[0] in ['[', '<'] or frame_file_path.find(' ') > -1 or (frame_file_path.find('\\') > -1) or (frame_file_path.find('/') == -1):
            raise UnsupportedFrameFilename('This path is not supported.')
        self.full_path = frame_file_path
        self.extension = get_extension(frame_file_path)
        if not self.extension:
            raise UnsupportedFrameFilename('It needs an extension.')
        if self.frame_type() == 'packaged':
            self._packaged_logic(frame_file_path)
        else:
            self._straight_path_logic(frame_file_path)

    def frame_type(self) -> str:
        if False:
            i = 10
            return i + 15
        type = 'packaged'
        if self.extension not in ['py']:
            type = 'straight_path'
        return type

    def _packaged_logic(self, frame_file_path: str) -> None:
        if False:
            print('Hello World!')
        (self.root, self.file_and_dir_path) = frame_file_path.split('/', 1)
        if self.file_and_dir_path.find('/') > -1:
            (self.dir_path, self.file_name) = self.file_and_dir_path.rsplit('/', 1)
        else:
            self.dir_path = ''
            self.file_name = self.file_and_dir_path

    def _straight_path_logic(self, frame_file_path: str) -> None:
        if False:
            while True:
                i = 10
        start_at_index = get_straight_path_prefix_end_index(frame_file_path)
        backslash_index = frame_file_path.find('/', start_at_index)
        (dir_path, self.file_name) = frame_file_path.rsplit('/', 1)
        self.root = frame_file_path[0:backslash_index]
        self.dir_path = dir_path.replace(self.root, '')
        self.file_and_dir_path = remove_straight_path_prefix(frame_file_path)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'FrameFilename: {self.full_path}'

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        return self.full_path == other.full_path

def stacktrace_buckets(stacktraces: List[str]) -> Dict[str, List[FrameFilename]]:
    if False:
        print('Hello World!')
    buckets: Dict[str, List[FrameFilename]] = {}
    for stacktrace_frame_file_path in stacktraces:
        try:
            frame_filename = FrameFilename(stacktrace_frame_file_path)
            bucket_key = frame_filename.root
            if not buckets.get(bucket_key):
                buckets[bucket_key] = []
            buckets[bucket_key].append(frame_filename)
        except UnsupportedFrameFilename:
            logger.info(f"Frame's filepath not supported: {stacktrace_frame_file_path}")
        except Exception:
            logger.exception('Unable to split stacktrace path into buckets')
    return buckets

class CodeMappingTreesHelper:

    def __init__(self, trees: Dict[str, RepoTree]):
        if False:
            for i in range(10):
                print('nop')
        self.trees = trees
        self.code_mappings: Dict[str, CodeMapping] = {}

    def process_stackframes(self, buckets: Dict[str, List[FrameFilename]]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'This processes all stackframes and returns if a new code mapping has been generated'
        reprocess = False
        for (stackframe_root, stackframes) in buckets.items():
            if not self.code_mappings.get(stackframe_root):
                for frame_filename in stackframes:
                    code_mapping = self._find_code_mapping(frame_filename)
                    if code_mapping:
                        reprocess = True
                        self.code_mappings[stackframe_root] = code_mapping
        return reprocess

    def generate_code_mappings(self, stacktraces: List[str]) -> List[CodeMapping]:
        if False:
            while True:
                i = 10
        'Generate code mappings based on the initial trees object and the list of stack traces'
        self.code_mappings = {}
        buckets: Dict[str, List[FrameFilename]] = stacktrace_buckets(stacktraces)
        while True:
            if not self.process_stackframes(buckets):
                break
        return list(self.code_mappings.values())

    def _find_code_mapping(self, frame_filename: FrameFilename) -> Union[CodeMapping, None]:
        if False:
            print('Hello World!')
        'Look for the file path through all the trees and generate code mappings for it'
        _code_mappings: List[CodeMapping] = []
        for repo_full_name in self.trees.keys():
            try:
                _code_mappings.extend(self._generate_code_mapping_from_tree(self.trees[repo_full_name], frame_filename))
            except NotImplementedError:
                logger.exception('Code mapping failed for module with no package name. Processing continues.')
            except Exception:
                logger.exception('Unexpected error. Processing continues.')
        if len(_code_mappings) == 0:
            logger.warning(f'No files matched for {frame_filename.full_path}')
            return None
        elif len(_code_mappings) > 1:
            logger.warning(f'More than one repo matched {frame_filename.full_path}')
            return None
        return _code_mappings[0]

    def list_file_matches(self, frame_filename: FrameFilename) -> List[Dict[str, str]]:
        if False:
            print('Hello World!')
        file_matches = []
        for repo_full_name in self.trees.keys():
            repo_tree = self.trees[repo_full_name]
            matches = [src_path for src_path in repo_tree.files if self._potential_match(src_path, frame_filename)]
            for file in matches:
                file_matches.append({'filename': file, 'repo_name': repo_tree.repo.name, 'repo_branch': repo_tree.repo.branch, 'stacktrace_root': f'{frame_filename.root}/', 'source_path': _get_code_mapping_source_path(file, frame_filename)})
        return file_matches

    def _normalized_stack_and_source_roots(self, stacktrace_root: str, source_path: str) -> Tuple[str, str]:
        if False:
            return 10
        if source_path == stacktrace_root:
            stacktrace_root = ''
            source_path = ''
        elif (without := remove_straight_path_prefix(stacktrace_root)) != stacktrace_root:
            start_index = get_straight_path_prefix_end_index(stacktrace_root)
            starts_with = stacktrace_root[:start_index]
            if source_path == without:
                stacktrace_root = starts_with
                source_path = ''
            elif source_path.rfind(f'/{without}'):
                stacktrace_root = starts_with
                source_path = source_path.replace(f'/{without}', '/')
        return (stacktrace_root, source_path)

    def _generate_code_mapping_from_tree(self, repo_tree: RepoTree, frame_filename: FrameFilename) -> List[CodeMapping]:
        if False:
            i = 10
            return i + 15
        matched_files = [src_path for src_path in repo_tree.files if self._potential_match(src_path, frame_filename)]
        if len(matched_files) != 1:
            return []
        stacktrace_root = f'{frame_filename.root}/'
        source_path = _get_code_mapping_source_path(matched_files[0], frame_filename)
        if frame_filename.frame_type() != 'packaged':
            (stacktrace_root, source_path) = self._normalized_stack_and_source_roots(stacktrace_root, source_path)
        return [CodeMapping(repo=repo_tree.repo, stacktrace_root=stacktrace_root, source_path=source_path)]

    def _matches_current_code_mappings(self, src_file: str, frame_filename: FrameFilename) -> bool:
        if False:
            print('Hello World!')
        return any((code_mapping.source_path for code_mapping in self.code_mappings.values() if src_file.startswith(f'{code_mapping.source_path}/')))

    def _potential_match_with_transformation(self, src_file: str, frame_filename: FrameFilename) -> bool:
        if False:
            while True:
                i = 10
        'Determine if the frame filename represents a source code file.\n\n        Languages like Python include the package name at the front of the frame_filename, thus, we need\n        to drop it before we try to match it.\n        '
        match = False
        split = src_file.split(f'/{frame_filename.file_and_dir_path}')
        if any(split) and len(split) > 1:
            match = split[0].rfind(f'/{frame_filename.root}') > -1 or split[0] == frame_filename.root
        return match

    def _potential_match_no_transformation(self, src_file: str, frame_filename: FrameFilename) -> bool:
        if False:
            return 10
        return src_file.rfind(frame_filename.file_and_dir_path) > -1

    def _potential_match(self, src_file: str, frame_filename: FrameFilename) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Tries to see if the stacktrace without the root matches the file from the\n        source code. Use existing code mappings to exclude some source files\n        '
        if self._matches_current_code_mappings(src_file, frame_filename):
            return False
        match = False
        if frame_filename.full_path.endswith('.py'):
            match = self._potential_match_with_transformation(src_file, frame_filename)
        else:
            match = self._potential_match_no_transformation(src_file, frame_filename)
        return match

def create_code_mapping(organization_integration: Union[OrganizationIntegration, RpcOrganizationIntegration], project: Project, code_mapping: CodeMapping) -> RepositoryProjectPathConfig:
    if False:
        i = 10
        return i + 15
    (repository, _) = Repository.objects.get_or_create(name=code_mapping.repo.name, organization_id=organization_integration.organization_id, defaults={'integration_id': organization_integration.integration_id})
    (new_code_mapping, created) = RepositoryProjectPathConfig.objects.update_or_create(project=project, stack_root=code_mapping.stacktrace_root, defaults={'repository': repository, 'organization_id': organization_integration.organization_id, 'integration_id': organization_integration.integration_id, 'organization_integration_id': organization_integration.id, 'source_root': code_mapping.source_path, 'default_branch': code_mapping.repo.branch, 'automatically_generated': True})
    if created:
        logger.info(f'Created a code mapping for project.slug={project.slug!r}, stack root: {code_mapping.stacktrace_root}')
    else:
        logger.info(f'Updated existing code mapping for project.slug={project.slug!r}, stack root: {code_mapping.stacktrace_root}')
    return new_code_mapping

def get_sorted_code_mapping_configs(project: Project) -> List[RepositoryProjectPathConfig]:
    if False:
        i = 10
        return i + 15
    '\n    Returns the code mapping config list for a project sorted based on precedence.\n    User generated code mappings are evaluated before Sentry generated code mappings.\n    Code mappings with more defined stack trace roots are evaluated before less defined stack trace\n    roots.\n\n    `project`: The project to get the list of sorted code mapping configs for\n    '
    configs = RepositoryProjectPathConfig.objects.filter(project=project, organization_integration_id__isnull=False)
    sorted_configs: list[RepositoryProjectPathConfig] = []
    try:
        for config in configs:
            inserted = False
            for (index, sorted_config) in enumerate(sorted_configs):
                if sorted_config.automatically_generated and (not config.automatically_generated) or (sorted_config.automatically_generated == config.automatically_generated and config.stack_root.startswith(sorted_config.stack_root)):
                    sorted_configs.insert(index, config)
                    inserted = True
                    break
            if not inserted:
                if config.automatically_generated:
                    sorted_configs.insert(len(sorted_configs), config)
                else:
                    sorted_configs.insert(0, config)
    except Exception:
        logger.exception('There was a failure sorting the code mappings')
    return sorted_configs

def _get_code_mapping_source_path(src_file: str, frame_filename: FrameFilename) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate the source code root for a code mapping. It always includes a last backslash'
    source_code_root = None
    if frame_filename.frame_type() == 'packaged':
        if frame_filename.dir_path != '':
            source_path = src_file.rsplit(frame_filename.dir_path)[0].rstrip('/')
            source_code_root = f'{source_path}/'
        elif frame_filename.root != '':
            source_code_root = src_file.rsplit(frame_filename.file_name)[0]
        else:
            raise NotImplementedError('We do not support top level files.')
    else:
        source_code_root = f'{src_file.replace(frame_filename.file_and_dir_path, remove_straight_path_prefix(frame_filename.root))}/'
    if source_code_root:
        assert source_code_root.endswith('/')
    return source_code_root