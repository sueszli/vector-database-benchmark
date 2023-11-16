""" Class to handle source code management repositories. """
import logging
import subprocess
logger = logging.getLogger(__name__)
try:
    import git
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False
HAS_GITPYTHON = False

class InvalidSCMError(Exception):
    """ Exception for when trying to access a repo of wrong type. """

    def __init__(self):
        if False:
            while True:
                i = 10
        Exception.__init__(self)

class SCMRepository(object):
    """ Base class to handle interactions with source code management systems. """
    handles_scm_type = '*'

    def __init__(self, path_to_repo, is_empty=False):
        if False:
            return 10
        self.path_to_repo = path_to_repo
        self.is_empty = is_empty

    def init_repo(self, path_to_repo=None, add_files=True):
        if False:
            for i in range(10):
                print('nop')
        ' Initialize the directory as a repository. Assumes the self.path_to_repo\n        (or path_to_repo, if specified) does *not* contain a valid repository.\n        If add_files is True, all files in this directory are added to version control.\n        Returns true if actually created a repo.\n        '
        if path_to_repo is not None:
            self.path_to_repo = path_to_repo
        return False

    def add_files(self, paths_to_files):
        if False:
            while True:
                i = 10
        ' Add a tuple or list of files to the current repository. '
        pass

    def add_file(self, path_to_file):
        if False:
            print('Hello World!')
        ' Add a file to the current repository. '
        self.add_files([path_to_file])

    def remove_files(self, paths_to_files):
        if False:
            for i in range(10):
                print('nop')
        ' Remove a tuple or list of files from the current repository. '
        pass

    def remove_file(self, path_to_file):
        if False:
            print('Hello World!')
        ' Remove a file from the current repository. '
        self.remove_files([path_to_file])

    def mark_files_updated(self, paths_to_files):
        if False:
            print('Hello World!')
        ' Mark a list of tuple of files as changed. '
        pass

    def mark_file_updated(self, path_to_file):
        if False:
            for i in range(10):
                print('nop')
        ' Mark a file as changed. '
        self.mark_files_updated([path_to_file])

    def is_active(self):
        if False:
            while True:
                i = 10
        ' Returns true if this repository manager is operating on an active, source-controlled directory. '
        return self.is_empty

    def get_gituser(self):
        if False:
            i = 10
            return i + 15
        ' Gets the git user '
        try:
            return subprocess.check_output('git config --global user.name', shell=True).strip().decode('utf-8')
        except (OSError, subprocess.CalledProcessError):
            return None

class GitManagerGitPython(object):
    """ Manage git through GitPython (preferred way). """

    def __init__(self, path_to_repo, init=False):
        if False:
            while True:
                i = 10
        if init:
            self.repo = git.Repo.init(path_to_repo, mkdir=False)
        else:
            try:
                self.repo = git.Repo(path_to_repo)
            except git.InvalidGitRepositoryError:
                self.repo = None
                raise InvalidSCMError
        self.index = self.repo.index

    def add_files(self, paths_to_files):
        if False:
            while True:
                i = 10
        ' Adds a tuple of files to the index of the current repository. '
        if self.repo is not None:
            self.index.add(paths_to_files)

    def remove_files(self, paths_to_files):
        if False:
            print('Hello World!')
        ' Removes a tuple of files from the index of the current repository. '
        if self.repo is not None:
            self.index.remove(paths_to_files)

class GitManagerShell(object):
    """ Call the git executable through a shell. """

    def __init__(self, path_to_repo, init=False, git_executable=None):
        if False:
            return 10
        self.path_to_repo = path_to_repo
        if git_executable is None:
            try:
                self.git_executable = subprocess.check_output('which git', shell=True).strip()
            except (OSError, subprocess.CalledProcessError):
                raise InvalidSCMError
        try:
            if init:
                subprocess.check_output([self.git_executable, 'init'])
            else:
                subprocess.check_output([self.git_executable, 'status'])
        except OSError:
            raise InvalidSCMError
        except subprocess.CalledProcessError:
            raise InvalidSCMError

    def add_files(self, paths_to_files):
        if False:
            return 10
        ' Adds a tuple of files to the index of the current repository. Does not commit. '
        subprocess.check_output([self.git_executable, 'add'] + list(paths_to_files))

    def remove_files(self, paths_to_files):
        if False:
            i = 10
            return i + 15
        ' Removes a tuple of files from the index of the current repository. Does not commit. '
        subprocess.check_output([self.git_executable, 'rm', '--cached'] + list(paths_to_files))

class GitRepository(SCMRepository):
    """ Specific to operating on git repositories. """
    handles_scm_type = 'git'

    def __init__(self, path_to_repo, is_empty=False):
        if False:
            return 10
        SCMRepository.__init__(self, path_to_repo, is_empty)
        if not is_empty:
            try:
                if HAS_GITPYTHON:
                    self.repo_manager = GitManagerGitPython(path_to_repo)
                else:
                    self.repo_manager = GitManagerShell(path_to_repo)
            except InvalidSCMError:
                self.repo_manager = None
        else:
            self.repo_manager = None

    def init_repo(self, path_to_repo=None, add_files=True):
        if False:
            for i in range(10):
                print('nop')
        ' Makes the directory in self.path_to_repo a git repo.\n        If add_file is True, all files in this dir are added to the index. '
        SCMRepository.init_repo(self, path_to_repo, add_files)
        if HAS_GITPYTHON:
            self.repo_manager = GitManagerGitPython(self.path_to_repo, init=True)
        else:
            self.repo_manager = GitManagerShell(self.path_to_repo, init=True)
        if add_files:
            self.add_files(('*',))
        return True

    def add_files(self, paths_to_files):
        if False:
            return 10
        ' Add a file to the current repository. Does not commit. '
        self.repo_manager.add_files(paths_to_files)

    def remove_files(self, paths_to_files):
        if False:
            while True:
                i = 10
        ' Remove a file from the current repository. Does not commit. '
        self.repo_manager.remove_files(paths_to_files)

    def mark_files_updated(self, paths_to_files):
        if False:
            while True:
                i = 10
        ' Mark a file as changed. Since this is git, same as adding new files. '
        self.add_files(paths_to_files)

    def is_active(self):
        if False:
            for i in range(10):
                print('nop')
        return self.repo_manager is not None

class SCMRepoFactory(object):
    """ Factory object to create the correct SCM class from the given options and dir. """

    def __init__(self, options, path_to_repo):
        if False:
            print('Hello World!')
        self.path_to_repo = path_to_repo
        self.options = options

    def make_active_scm_manager(self):
        if False:
            i = 10
            return i + 15
        ' Returns a valid, usable object of type SCMRepository. '
        if self.options.scm_mode == 'no':
            return SCMRepository(self.path_to_repo)
        for glbl in list(globals().values()):
            try:
                if issubclass(glbl, SCMRepository):
                    the_scm = glbl(self.path_to_repo)
                    if the_scm.is_active():
                        logger.info(f'Found SCM of type: {the_scm.handles_scm_type}')
                        return the_scm
            except (TypeError, AttributeError, InvalidSCMError):
                pass
        if self.options == 'yes':
            return None
        return SCMRepository(self.path_to_repo)

    def make_empty_scm_manager(self, scm_type='git'):
        if False:
            return 10
        ' Returns a valid, usable object of type SCMRepository for an uninitialized dir. '
        if self.options.scm_mode == 'no':
            return SCMRepository(self.path_to_repo)
        for glbl in list(globals().values()):
            try:
                if issubclass(glbl, SCMRepository):
                    if glbl.handles_scm_type == scm_type:
                        return glbl(self.path_to_repo, is_empty=True)
            except (TypeError, AttributeError, InvalidSCMError):
                pass
        if self.options == 'yes':
            return None
        return SCMRepository(self.path_to_repo)