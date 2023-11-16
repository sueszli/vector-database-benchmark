"""Load, parse and make available Greasemonkey scripts."""
import re
import os
import json
import fnmatch
import functools
import glob
import textwrap
import dataclasses
from typing import cast, List, Sequence, Tuple, Optional
from qutebrowser.qt.core import pyqtSignal, QObject, QUrl
from qutebrowser.utils import log, standarddir, jinja, objreg, utils, javascript, urlmatch, version, usertypes, message
from qutebrowser.api import cmdutils
from qutebrowser.browser import downloads
from qutebrowser.misc import objects
gm_manager = cast('GreasemonkeyManager', None)

def _scripts_dirs():
    if False:
        return 10
    'Get the directory of the scripts.'
    return [os.path.join(standarddir.data(), 'greasemonkey'), os.path.join(standarddir.config(), 'greasemonkey')]

class GreasemonkeyScript:
    """Container class for userscripts, parses metadata blocks."""

    def __init__(self, properties, code, filename=None):
        if False:
            return 10
        self._code = code
        self.includes: Sequence[str] = []
        self.matches: Sequence[str] = []
        self.excludes: Sequence[str] = []
        self.requires: Sequence[str] = []
        self.description = None
        self.namespace = None
        self.run_at = None
        self.script_meta = None
        self.runs_on_sub_frames = True
        self.jsworld = 'main'
        self.name = ''
        self.dedup_suffix = 1
        for (name, value) in properties:
            if name == 'name':
                self.name = value
            elif name == 'namespace':
                self.namespace = value
            elif name == 'description':
                self.description = value
            elif name == 'include':
                self.includes.append(value)
            elif name == 'match':
                self.matches.append(value)
            elif name in ['exclude', 'exclude_match']:
                self.excludes.append(value)
            elif name == 'run-at':
                self.run_at = value
            elif name == 'noframes':
                self.runs_on_sub_frames = False
            elif name == 'require':
                self.requires.append(value)
            elif name == 'qute-js-world':
                self.jsworld = value
        if not self.name:
            if filename:
                self.name = filename
            else:
                raise ValueError('@name key required or pass filename to init.')
    HEADER_REGEX = '// ==UserScript==|\\n+// ==/UserScript==\\n'
    PROPS_REGEX = '// @(?P<prop>[^\\s]+)\\s*(?P<val>.*)'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.name

    def full_name(self) -> str:
        if False:
            while True:
                i = 10
        'Get the full name of this script.\n\n        This includes a GM- prefix, its namespace (if any) and deduplication\n        counter suffix, if set.\n        '
        parts = ['GM-']
        if self.namespace is not None:
            parts += [self.namespace, '/']
        parts.append(self.name)
        if self.dedup_suffix > 1:
            parts.append(f'-{self.dedup_suffix}')
        return ''.join(parts)

    @classmethod
    def parse(cls, source, filename=None):
        if False:
            for i in range(10):
                print('nop')
        'GreasemonkeyScript factory.\n\n        Takes a userscript source and returns a GreasemonkeyScript.\n        Parses the Greasemonkey metadata block, if present, to fill out\n        attributes.\n        '
        matches = re.split(cls.HEADER_REGEX, source, maxsplit=2)
        try:
            (_head, props, _code) = matches
        except ValueError:
            props = ''
        script = cls(re.findall(cls.PROPS_REGEX, props), source, filename=filename)
        script.script_meta = props
        if not script.includes and (not script.matches):
            script.includes = ['*']
        return script

    def needs_document_end_workaround(self):
        if False:
            while True:
                i = 10
        'Check whether to force @run-at document-end.\n\n        This needs to be done on QtWebEngine for known-broken scripts.\n\n        On Qt 5.12, accessing the DOM isn\'t possible with "@run-at\n        document-start". It was documented to be impossible before, but seems\n        to work fine.\n\n        However, some scripts do DOM access with "@run-at document-start". Fix\n        those by forcing them to use document-end instead.\n        '
        if objects.backend == usertypes.Backend.QtWebKit:
            return False
        assert objects.backend == usertypes.Backend.QtWebEngine, objects.backend
        broken_scripts = [('http://userstyles.org', None), ('https://github.com/ParticleCore', 'Iridium')]
        return any((self._matches_id(namespace=namespace, name=name) for (namespace, name) in broken_scripts))

    def _matches_id(self, *, namespace, name):
        if False:
            for i in range(10):
                print('nop')
        'Check if this script matches the given namespace/name.\n\n        Both namespace and name can be None in order to match any script.\n        '
        matches_namespace = namespace is None or self.namespace == namespace
        matches_name = name is None or self.name == name
        return matches_namespace and matches_name

    def code(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the processed JavaScript code of this script.\n\n        Adorns the source code with GM_* methods for Greasemonkey\n        compatibility and wraps it in an IIFE to hide it within a\n        lexical scope. Note that this means line numbers in your\n        browser's debugger/inspector will not match up to the line\n        numbers in the source script directly.\n        "
        use_proxy = not (objects.backend == usertypes.Backend.QtWebKit and version.qWebKitVersion() == '602.1')
        template = jinja.js_environment.get_template('greasemonkey_wrapper.js')
        return template.render(scriptName=javascript.string_escape('/'.join([self.namespace or '', self.name])), scriptInfo=self._meta_json(), scriptMeta=javascript.string_escape(self.script_meta or ''), scriptSource=self._code, use_proxy=use_proxy)

    def _meta_json(self):
        if False:
            return 10
        return json.dumps({'name': self.name, 'description': self.description, 'matches': self.matches, 'includes': self.includes, 'excludes': self.excludes, 'run-at': self.run_at})

    def add_required_script(self, source):
        if False:
            return 10
        'Add the source of a required script to this script.'
        self._code = '\n'.join([textwrap.indent(source, '    '), self._code])

@dataclasses.dataclass
class MatchingScripts:
    """All userscripts registered to run on a particular url."""
    url: QUrl
    start: List[GreasemonkeyScript] = dataclasses.field(default_factory=list)
    end: List[GreasemonkeyScript] = dataclasses.field(default_factory=list)
    idle: List[GreasemonkeyScript] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class LoadResults:
    """The results of loading all Greasemonkey scripts."""
    successful: List[GreasemonkeyScript] = dataclasses.field(default_factory=list)
    errors: List[Tuple[str, str]] = dataclasses.field(default_factory=list)

    def successful_str(self) -> str:
        if False:
            return 10
        'Get a string with all successfully loaded scripts.\n\n        This can be used e.g. for a message.info() call.\n        '
        if not self.successful:
            return 'No Greasemonkey scripts loaded'
        names = '\n'.join((str(script) for script in sorted(self.successful, key=str)))
        return f'Loaded Greasemonkey scripts:\n\n{names}'

    def error_str(self) -> Optional[str]:
        if False:
            return 10
        'Get a string with all errors during script loading.\n\n        This can be used e.g. for a message.error() call.\n        If there were no errors, None is returned.\n        '
        if not self.errors:
            return None
        lines = '\n'.join((f'{script}: {error}' for (script, error) in sorted(self.errors)))
        return f'Greasemonkey scripts failed to load:\n\n{lines}'

class GreasemonkeyMatcher:
    """Check whether scripts should be loaded for a given URL."""
    GREASEABLE_SCHEMES = ['http', 'https', 'ftp', 'file']

    def __init__(self, url):
        if False:
            print('Hello World!')
        self._url = url
        self._url_string = url.toString(QUrl.ComponentFormattingOption.FullyEncoded)
        self.is_greaseable = url.scheme() in self.GREASEABLE_SCHEMES

    def _match_pattern(self, pattern):
        if False:
            i = 10
            return i + 15
        if pattern.startswith('/') and pattern.endswith('/'):
            matches = re.search(pattern[1:-1], self._url_string, flags=re.I)
            return matches is not None
        return fnmatch.fnmatch(self._url_string, pattern)

    def matches(self, script):
        if False:
            print('Hello World!')
        'Check whether the URL matches filtering rules of the script.'
        assert self.is_greaseable
        matching_includes = any((self._match_pattern(pat) for pat in script.includes))
        matching_match = any((urlmatch.UrlPattern(pat).matches(self._url) for pat in script.matches))
        matching_excludes = any((self._match_pattern(pat) for pat in script.excludes))
        return (matching_includes or matching_match) and (not matching_excludes)

class GreasemonkeyManager(QObject):
    """Manager of userscripts and a Greasemonkey compatible environment.

    Signals:
        scripts_reloaded: Emitted when scripts are reloaded from disk.
            Any cached or already-injected scripts should be
            considered obsolete.
    """
    scripts_reloaded = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self._run_start: List[GreasemonkeyScript] = []
        self._run_end: List[GreasemonkeyScript] = []
        self._run_idle: List[GreasemonkeyScript] = []
        self._in_progress_dls: List[downloads.AbstractDownloadItem] = []

    def load_scripts(self, *, force: bool=False) -> LoadResults:
        if False:
            for i in range(10):
                print('nop')
        "Re-read Greasemonkey scripts from disk.\n\n        The scripts are read from a 'greasemonkey' subdirectory in\n        qutebrowser's data directory (see `:version`).\n\n        Args:\n            force: For any scripts that have required dependencies,\n                   re-download them.\n\n        Return:\n            A LoadResults object describing the outcome.\n        "
        self._run_start = []
        self._run_end = []
        self._run_idle = []
        successful = []
        errors = []
        for scripts_dir in _scripts_dirs():
            scripts_dir = os.path.abspath(scripts_dir)
            log.greasemonkey.debug('Reading scripts from: {}'.format(scripts_dir))
            for script_filename in glob.glob(os.path.join(scripts_dir, '*.js')):
                script_path = os.path.join(scripts_dir, script_filename)
                try:
                    with open(script_path, encoding='utf-8-sig') as script_file:
                        script = GreasemonkeyScript.parse(script_file.read(), script_filename)
                        assert script.name, script
                        self.add_script(script, force)
                        successful.append(script)
                except OSError as e:
                    errors.append((os.path.basename(script_filename), str(e)))
        self.scripts_reloaded.emit()
        return LoadResults(successful=successful, errors=errors)

    def add_script(self, script, force=False):
        if False:
            return 10
        'Add a GreasemonkeyScript to this manager.\n\n        Args:\n            script: The GreasemonkeyScript to add.\n            force: Fetch and overwrite any dependencies which are\n                   already locally cached.\n        '
        if script.requires:
            log.greasemonkey.debug(f'Deferring script until requirements are fulfilled: {script}')
            self._get_required_scripts(script, force)
        else:
            self._add_script(script)

    def _add_script(self, script):
        if False:
            for i in range(10):
                print('nop')
        if script.run_at == 'document-start':
            self._run_start.append(script)
        elif script.run_at == 'document-end':
            self._run_end.append(script)
        elif script.run_at == 'document-idle':
            self._run_idle.append(script)
        else:
            if script.run_at:
                log.greasemonkey.warning(f'Script {script} has invalid run-at defined, defaulting to document-end')
            self._run_end.append(script)
        log.greasemonkey.debug(f'Loaded script: {script}')

    def _required_url_to_file_path(self, url):
        if False:
            while True:
                i = 10
        requires_dir = os.path.join(_scripts_dirs()[0], 'requires')
        if not os.path.exists(requires_dir):
            os.mkdir(requires_dir)
        return os.path.join(requires_dir, utils.sanitize_filename(url))

    def _on_required_download_finished(self, script, download):
        if False:
            print('Hello World!')
        self._in_progress_dls.remove(download)
        if not self._add_script_with_requires(script):
            log.greasemonkey.debug(f'Finished download {download.basename} for script {script} but some requirements are still pending')

    def _add_script_with_requires(self, script, quiet=False):
        if False:
            return 10
        'Add a script with pending downloads to this GreasemonkeyManager.\n\n        Specifically a script that has dependencies specified via an\n        `@require` rule.\n\n        Args:\n            script: The GreasemonkeyScript to add.\n            quiet: True to suppress the scripts_reloaded signal after\n                   adding `script`.\n        Returns: True if the script was added, False if there are still\n                 dependencies being downloaded.\n        '
        for dl in self._in_progress_dls:
            if dl.requested_url in script.requires:
                return False
        for url in reversed(script.requires):
            target_path = self._required_url_to_file_path(url)
            log.greasemonkey.debug(f'Adding required script for {script} to IIFE: {url}')
            with open(target_path, encoding='utf8') as f:
                script.add_required_script(f.read())
        self._add_script(script)
        if not quiet:
            self.scripts_reloaded.emit()
        return True

    def _get_required_scripts(self, script, force=False):
        if False:
            i = 10
            return i + 15
        required_dls = [(url, self._required_url_to_file_path(url)) for url in script.requires]
        if not force:
            required_dls = [(url, path) for (url, path) in required_dls if not os.path.exists(path)]
        if not required_dls:
            self._add_script_with_requires(script, quiet=True)
            return
        download_manager = objreg.get('qtnetwork-download-manager')
        for (url, target_path) in required_dls:
            target = downloads.FileDownloadTarget(target_path, force_overwrite=True)
            download = download_manager.get(QUrl(url), target=target, auto_remove=True)
            download.requested_url = url
            self._in_progress_dls.append(download)
            if download.successful:
                self._on_required_download_finished(script, download)
            else:
                download.finished.connect(functools.partial(self._on_required_download_finished, script, download))

    def scripts_for(self, url):
        if False:
            while True:
                i = 10
        'Fetch scripts that are registered to run for url.\n\n        returns a tuple of lists of scripts meant to run at (document-start,\n        document-end, document-idle)\n        '
        matcher = GreasemonkeyMatcher(url)
        if not matcher.is_greaseable:
            return MatchingScripts(url, [], [], [])
        return MatchingScripts(url=url, start=[script for script in self._run_start if matcher.matches(script)], end=[script for script in self._run_end if matcher.matches(script)], idle=[script for script in self._run_idle if matcher.matches(script)])

    def all_scripts(self):
        if False:
            print('Hello World!')
        'Return all scripts found in the configured script directory.'
        return self._run_start + self._run_end + self._run_idle

@cmdutils.register()
def greasemonkey_reload(force: bool=False, quiet: bool=False) -> None:
    if False:
        return 10
    "Re-read Greasemonkey scripts from disk.\n\n    The scripts are read from a 'greasemonkey' subdirectory in\n    qutebrowser's data or config directories (see `:version`).\n\n    Args:\n        force: For any scripts that have required dependencies,\n                re-download them.\n        quiet: Suppress message after loading scripts.\n    "
    result = gm_manager.load_scripts(force=force)
    if not quiet:
        message.info(result.successful_str())
    errors = result.error_str()
    if errors is not None:
        message.error(errors)

def init():
    if False:
        print('Hello World!')
    'Initialize Greasemonkey support.'
    global gm_manager
    gm_manager = GreasemonkeyManager()
    result = gm_manager.load_scripts()
    errors = result.error_str()
    if errors is not None:
        message.error(errors)
    for scripts_dir in _scripts_dirs():
        try:
            os.mkdir(scripts_dir)
        except FileExistsError:
            pass