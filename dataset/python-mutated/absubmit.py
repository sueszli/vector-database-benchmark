"""Calculate acoustic information and submit to AcousticBrainz.
"""
import errno
import hashlib
import json
import os
import subprocess
import tempfile
from distutils.spawn import find_executable
import requests
from beets import plugins, ui, util
PROBE_FIELD = 'mood_acoustic'

class ABSubmitError(Exception):
    """Raised when failing to analyse file with extractor."""

def call(args):
    if False:
        return 10
    'Execute the command and return its output.\n\n    Raise a AnalysisABSubmitError on failure.\n    '
    try:
        return util.command_output(args).stdout
    except subprocess.CalledProcessError as e:
        raise ABSubmitError('{} exited with status {}'.format(args[0], e.returncode))

class AcousticBrainzSubmitPlugin(plugins.BeetsPlugin):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._log.warning('This plugin is deprecated.')
        self.config.add({'extractor': '', 'force': False, 'pretend': False, 'base_url': ''})
        self.extractor = self.config['extractor'].as_str()
        if self.extractor:
            self.extractor = util.normpath(self.extractor)
            if not os.path.isfile(self.extractor):
                raise ui.UserError('Extractor command does not exist: {0}.'.format(self.extractor))
        else:
            self.extractor = 'streaming_extractor_music'
            try:
                call([self.extractor])
            except OSError:
                raise ui.UserError('No extractor command found: please install the extractor binary from https://essentia.upf.edu/')
            except ABSubmitError:
                pass
            self.extractor = find_executable(self.extractor)
        self.extractor_sha = hashlib.sha1()
        with open(self.extractor, 'rb') as extractor:
            self.extractor_sha.update(extractor.read())
        self.extractor_sha = self.extractor_sha.hexdigest()
        self.url = ''
        base_url = self.config['base_url'].as_str()
        if base_url:
            if not base_url.startswith('http'):
                raise ui.UserError('AcousticBrainz server base URL must start with an HTTP scheme')
            elif base_url[-1] != '/':
                base_url = base_url + '/'
            self.url = base_url + '{mbid}/low-level'

    def commands(self):
        if False:
            i = 10
            return i + 15
        cmd = ui.Subcommand('absubmit', help='calculate and submit AcousticBrainz analysis')
        cmd.parser.add_option('-f', '--force', dest='force_refetch', action='store_true', default=False, help='re-download data when already present')
        cmd.parser.add_option('-p', '--pretend', dest='pretend_fetch', action='store_true', default=False, help='pretend to perform action, but show only files which would be processed')
        cmd.func = self.command
        return [cmd]

    def command(self, lib, opts, args):
        if False:
            for i in range(10):
                print('nop')
        if not self.url:
            raise ui.UserError('This plugin is deprecated since AcousticBrainz no longer accepts new submissions. See the base_url configuration option.')
        else:
            items = lib.items(ui.decargs(args))
            self.opts = opts
            util.par_map(self.analyze_submit, items)

    def analyze_submit(self, item):
        if False:
            while True:
                i = 10
        analysis = self._get_analysis(item)
        if analysis:
            self._submit_data(item, analysis)

    def _get_analysis(self, item):
        if False:
            i = 10
            return i + 15
        mbid = item['mb_trackid']
        if not self.opts.force_refetch and (not self.config['force']):
            if item.get(PROBE_FIELD):
                return None
        if not mbid:
            self._log.info('Not analysing {}, missing musicbrainz track id.', item)
            return None
        if self.opts.pretend_fetch or self.config['pretend']:
            self._log.info('pretend action - extract item: {}', item)
            return None
        (tmp_file, filename) = tempfile.mkstemp(suffix='.json')
        try:
            os.close(tmp_file)
            try:
                call([self.extractor, util.syspath(item.path), filename])
            except ABSubmitError as e:
                self._log.warning('Failed to analyse {item} for AcousticBrainz: {error}', item=item, error=e)
                return None
            with open(filename) as tmp_file:
                analysis = json.load(tmp_file)
            analysis['metadata']['version']['essentia_build_sha'] = self.extractor_sha
            return analysis
        finally:
            try:
                os.remove(filename)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise

    def _submit_data(self, item, data):
        if False:
            for i in range(10):
                print('nop')
        mbid = item['mb_trackid']
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.url.format(mbid=mbid), json=data, headers=headers)
        if response.status_code != 200:
            try:
                message = response.json()['message']
            except (ValueError, KeyError) as e:
                message = f'unable to get error message: {e}'
            self._log.error('Failed to submit AcousticBrainz analysis of {item}: {message}).', item=item, message=message)
        else:
            self._log.debug('Successfully submitted AcousticBrainz analysis for {}.', item)