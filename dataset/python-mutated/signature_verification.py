"""Interface between conda-content-trust and conda."""
import json
import warnings
from functools import lru_cache
from glob import glob
from logging import getLogger
from os import makedirs
from os.path import basename, exists, isdir, join
try:
    from conda_content_trust.authentication import verify_delegation, verify_root
    from conda_content_trust.common import SignatureError, load_metadata_from_file, write_metadata_to_file
    from conda_content_trust.signing import wrap_as_signable
except ImportError:

    class SignatureError(Exception):
        pass
from ..base.context import context
from ..common.url import join_url
from ..gateways.connection import HTTPError, InsecureRequestWarning
from ..gateways.connection.session import get_session
from .constants import INITIAL_TRUST_ROOT, KEY_MGR_FILE
log = getLogger(__name__)

class _SignatureVerification:

    @property
    @lru_cache(maxsize=None)
    def enabled(self):
        if False:
            return 10
        if not context.extra_safety_checks:
            return False
        if not context.signing_metadata_url_base:
            log.warn('metadata signature verification requested, but no metadata URL base has not been specified.')
            return False
        try:
            import conda_content_trust
        except ImportError:
            log.warn('metadata signature verification requested, but `conda-content-trust` is not installed.')
            return False
        if not isdir(context.av_data_dir):
            log.info('creating directory for artifact verification metadata')
            makedirs(context.av_data_dir)
        if self.trusted_root is None:
            log.warn('could not find trusted_root data for metadata signature verification')
            return False
        if self.key_mgr is None:
            log.warn('could not find key_mgr data for metadata signature verification')
            return False
        return True

    @property
    @lru_cache(maxsize=None)
    def trusted_root(self):
        if False:
            i = 10
            return i + 15
        trusted = INITIAL_TRUST_ROOT
        for path in sorted(glob(join(context.av_data_dir, '[0-9]*.root.json')), reverse=True):
            try:
                int(basename(path).split('.')[0])
            except ValueError:
                pass
            else:
                log.info(f'Loading root metadata from {path}.')
                trusted = load_metadata_from_file(path)
                break
        else:
            log.debug(f'No root metadata in {context.av_data_dir}. Using built-in root metadata.')
        more_signatures = True
        while more_signatures:
            fname = f"{trusted['signed']['version'] + 1}.root.json"
            path = join(context.av_data_dir, fname)
            try:
                untrusted = self._fetch_channel_signing_data(context.signing_metadata_url_base, fname)
                verify_root(trusted, untrusted)
            except HTTPError as err:
                if err.response.status_code != 404:
                    log.error(err)
                more_signatures = False
            except Exception as err:
                log.error(err)
                more_signatures = False
            else:
                trusted = untrusted
                write_metadata_to_file(trusted, path)
        return trusted

    @property
    @lru_cache(maxsize=None)
    def key_mgr(self):
        if False:
            return 10
        trusted = None
        fname = KEY_MGR_FILE
        path = join(context.av_data_dir, fname)
        try:
            untrusted = self._fetch_channel_signing_data(context.signing_metadata_url_base, KEY_MGR_FILE)
            verify_delegation('key_mgr', untrusted, self.trusted_root)
        except (ConnectionError, HTTPError) as err:
            log.warn(err)
        except Exception as err:
            raise
            log.error(err)
        else:
            trusted = untrusted
            write_metadata_to_file(trusted, path)
        if not trusted and exists(path):
            trusted = load_metadata_from_file(path)
        return trusted

    def _fetch_channel_signing_data(self, signing_data_url, filename, etag=None, mod_stamp=None):
        if False:
            for i in range(10):
                print('nop')
        session = get_session(signing_data_url)
        if not context.ssl_verify:
            warnings.simplefilter('ignore', InsecureRequestWarning)
        headers = {'Accept-Encoding': 'gzip, deflate, compress, identity', 'Content-Type': 'application/json'}
        if etag:
            headers['If-None-Match'] = etag
        if mod_stamp:
            headers['If-Modified-Since'] = mod_stamp
        saved_token_setting = context.add_anaconda_token
        try:
            context.add_anaconda_token = False
            resp = session.get(join_url(signing_data_url, filename), headers=headers, proxies=session.proxies, auth=None, timeout=(context.remote_connect_timeout_secs, context.remote_read_timeout_secs))
            resp.raise_for_status()
        finally:
            context.add_anaconda_token = saved_token_setting
        try:
            return resp.json()
        except json.decoder.JSONDecodeError as err:
            raise ValueError(f'Invalid JSON returned from {signing_data_url}/{filename}')

    def __call__(self, info, fn, signatures):
        if False:
            for i in range(10):
                print('nop')
        if not self.enabled or fn not in signatures:
            return
        envelope = wrap_as_signable(info)
        envelope['signatures'] = signatures[fn]
        try:
            verify_delegation('pkg_mgr', envelope, self.key_mgr)
        except SignatureError:
            log.warn(f'invalid signature for {fn}')
            status = '(WARNING: metadata signature verification failed)'
        else:
            status = '(INFO: package metadata is signed by Anaconda and trusted)'
        info['metadata_signature_status'] = status
signature_verification = _SignatureVerification()