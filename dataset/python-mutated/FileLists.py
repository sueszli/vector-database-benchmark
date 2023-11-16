from __future__ import absolute_import
from .S3 import S3
from .Config import Config
from .S3Uri import S3Uri
from .FileDict import FileDict
from .BaseUtils import dateS3toUnix, dateRFC822toUnix, s3path
from .Utils import unicodise, deunicodise, deunicodise_s, replace_nonprintables
from .Exceptions import ParameterError
from .HashCache import HashCache
from logging import debug, info, warning
import os
import sys
import glob
import re
import errno
import io
from stat import S_ISDIR
PY3 = sys.version_info >= (3, 0)
__all__ = ['fetch_local_list', 'fetch_remote_list', 'compare_filelists']

def _os_walk_unicode(top):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reimplementation of python's os.walk to nicely support unicode in input as in output.\n    "
    try:
        names = os.listdir(deunicodise(top))
    except Exception:
        return
    (dirs, nondirs) = ([], [])
    for name in names:
        name = unicodise(name)
        if os.path.isdir(deunicodise(os.path.join(top, name))):
            if not handle_exclude_include_walk_dir(top, name):
                dirs.append(name)
        else:
            nondirs.append(name)
    yield (top, dirs, nondirs)
    for name in dirs:
        new_path = os.path.join(top, name)
        if not os.path.islink(deunicodise(new_path)):
            for x in _os_walk_unicode(new_path):
                yield x

def handle_exclude_include_walk_dir(root, dirname):
    if False:
        while True:
            i = 10
    '\n    Should this root/dirname directory be excluded? (otherwise included by default)\n    Exclude dir matches in the current directory\n    This prevents us from recursing down trees we know we want to ignore\n    return True for excluding, and False for including\n    '
    cfg = Config()
    directory_patterns = (u'/)$', u'/)\\Z', u'\\/$', u'\\/\\Z(?ms)')
    d = os.path.join(root, dirname, '')
    debug(u"CHECK: '%s'" % d)
    excluded = False
    for r in cfg.exclude:
        if not any((r.pattern.endswith(dp) for dp in directory_patterns)):
            continue
        if r.search(d):
            excluded = True
            debug(u"EXCL-MATCH: '%s'" % cfg.debug_exclude[r])
            break
    if excluded:
        for r in cfg.include:
            if not any((r.pattern.endswith(dp) for dp in directory_patterns)):
                continue
            debug(u"INCL-TEST: '%s' ~ %s" % (d, r.pattern))
            if r.search(d):
                excluded = False
                debug(u"INCL-MATCH: '%s'" % cfg.debug_include[r])
                break
    if excluded:
        debug(u"EXCLUDE: '%s'" % d)
    else:
        debug(u"PASS: '%s'" % d)
    return excluded

def _fswalk_follow_symlinks(path):
    if False:
        print('Hello World!')
    '\n    Walk filesystem, following symbolic links (but without recursion), on python2.4 and later\n\n    If a symlink directory loop is detected, emit a warning and skip.\n    E.g.: dir1/dir2/sym-dir -> ../dir2\n    '
    assert os.path.isdir(deunicodise(path))
    walkdirs = set([path])
    for (dirpath, dirnames, filenames) in _os_walk_unicode(path):
        real_dirpath = unicodise(os.path.realpath(deunicodise(dirpath)))
        for dirname in dirnames:
            current = os.path.join(dirpath, dirname)
            real_current = unicodise(os.path.realpath(deunicodise(current)))
            if os.path.islink(deunicodise(current)):
                if real_dirpath == real_current or real_dirpath.startswith(real_current + os.path.sep):
                    warning('Skipping recursively symlinked directory %s' % dirname)
                else:
                    walkdirs.add(current)
    for walkdir in walkdirs:
        for (dirpath, dirnames, filenames) in _os_walk_unicode(walkdir):
            yield (dirpath, dirnames, filenames)

def _fswalk_no_symlinks(path):
    if False:
        while True:
            i = 10
    '\n    Directory tree generator\n\n    path (str) is the root of the directory tree to walk\n    '
    for (dirpath, dirnames, filenames) in _os_walk_unicode(path):
        yield (dirpath, dirnames, filenames)

def filter_exclude_include(src_list):
    if False:
        return 10
    debug(u'Applying --exclude/--include')
    cfg = Config()
    exclude_list = FileDict(ignore_case=False)
    for file in src_list.keys():
        debug(u"CHECK: '%s'" % file)
        excluded = False
        for r in cfg.exclude:
            if r.search(file):
                excluded = True
                debug(u"EXCL-MATCH: '%s'" % cfg.debug_exclude[r])
                break
        if excluded:
            for r in cfg.include:
                if r.search(file):
                    excluded = False
                    debug(u"INCL-MATCH: '%s'" % cfg.debug_include[r])
                    break
        if excluded:
            debug(u"EXCLUDE: '%s'" % file)
            exclude_list[file] = src_list[file]
            del src_list[file]
            continue
        else:
            debug(u"PASS: '%s'" % file)
    return (src_list, exclude_list)

def _get_filelist_from_file(cfg, local_path):
    if False:
        print('Hello World!')

    def _append(d, key, value):
        if False:
            i = 10
            return i + 15
        if key not in d:
            d[key] = [value]
        else:
            d[key].append(value)
    filelist = {}
    for fname in cfg.files_from:
        try:
            f = None
            if fname == u'-':
                f = io.open(sys.stdin.fileno(), mode='r', closefd=False)
            else:
                try:
                    f = io.open(deunicodise(fname), mode='r')
                except IOError as e:
                    warning(u'--files-from input file %s could not be opened for reading (%s), skipping.' % (fname, e.strerror))
                    continue
            for line in f:
                line = unicodise(line).strip()
                line = os.path.normpath(os.path.join(local_path, line))
                dirname = unicodise(os.path.dirname(deunicodise(line)))
                basename = unicodise(os.path.basename(deunicodise(line)))
                _append(filelist, dirname, basename)
        finally:
            if f:
                f.close()
    result = []
    for key in sorted(filelist):
        values = filelist[key]
        values.sort()
        result.append((key, [], values))
    return result

def fetch_local_list(args, is_src=False, recursive=None, with_dirs=False):
    if False:
        print('Hello World!')

    def _fetch_local_list_info(loc_list):
        if False:
            return 10
        len_loc_list = len(loc_list)
        total_size = 0
        info(u'Running stat() and reading/calculating MD5 values on %d files, this may take some time...' % len_loc_list)
        counter = 0
        for relative_file in loc_list:
            counter += 1
            if counter % 1000 == 0:
                info(u'[%d/%d]' % (counter, len_loc_list))
            if relative_file == '-':
                continue
            loc_list_item = loc_list[relative_file]
            full_name = loc_list_item['full_name']
            is_dir = loc_list_item['is_dir']
            try:
                sr = os.stat_result(os.stat(deunicodise(full_name)))
            except OSError as e:
                if e.errno == errno.ENOENT:
                    continue
                else:
                    raise
            if is_dir:
                size = 0
            else:
                size = sr.st_size
            loc_list[relative_file].update({'size': size, 'mtime': sr.st_mtime, 'dev': sr.st_dev, 'inode': sr.st_ino, 'uid': sr.st_uid, 'gid': sr.st_gid, 'sr': sr})
            total_size += sr.st_size
            if is_dir:
                continue
            if 'md5' in cfg.sync_checks:
                md5 = cache.md5(sr.st_dev, sr.st_ino, sr.st_mtime, sr.st_size)
                if md5 is None:
                    try:
                        md5 = loc_list.get_md5(relative_file)
                    except IOError:
                        continue
                    cache.add(sr.st_dev, sr.st_ino, sr.st_mtime, sr.st_size, md5)
                loc_list.record_hardlink(relative_file, sr.st_dev, sr.st_ino, md5, sr.st_size)
        return total_size

    def _get_filelist_local(loc_list, local_uri, cache, with_dirs):
        if False:
            while True:
                i = 10
        info(u'Compiling list of local files...')
        if local_uri.basename() == '-':
            try:
                uid = os.geteuid()
                gid = os.getegid()
            except Exception:
                uid = 0
                gid = 0
            loc_list['-'] = {'full_name': '-', 'size': -1, 'mtime': -1, 'uid': uid, 'gid': gid, 'dev': 0, 'inode': 0, 'is_dir': False}
            return (loc_list, True)
        if local_uri.isdir():
            local_base = local_uri.basename()
            local_path = local_uri.path()
            if is_src and len(cfg.files_from):
                filelist = _get_filelist_from_file(cfg, local_path)
                single_file = False
            else:
                if cfg.follow_symlinks:
                    filelist = _fswalk_follow_symlinks(local_path)
                else:
                    filelist = _fswalk_no_symlinks(local_path)
                single_file = False
        else:
            local_base = ''
            local_path = local_uri.dirname()
            filelist = [(local_path, [], [local_uri.basename()])]
            single_file = True
        for (root, dirs, files) in filelist:
            rel_root = root.replace(local_path, local_base, 1)
            if not with_dirs:
                iter_elements = ((files, False),)
            else:
                iter_elements = ((dirs, True), (files, False))
            for (elements, is_dir) in iter_elements:
                for f in elements:
                    full_name = os.path.join(root, f)
                    if not is_dir and (not os.path.isfile(deunicodise(full_name))):
                        if os.path.exists(deunicodise(full_name)):
                            warning(u'Skipping over non regular file: %s' % full_name)
                        continue
                    if os.path.islink(deunicodise(full_name)):
                        if not cfg.follow_symlinks:
                            warning(u'Skipping over symbolic link: %s' % full_name)
                            continue
                    relative_file = os.path.join(rel_root, f)
                    if os.path.sep != '/':
                        relative_file = '/'.join(relative_file.split(os.path.sep))
                    if cfg.urlencoding_mode == 'normal':
                        relative_file = replace_nonprintables(relative_file)
                    if relative_file.startswith('./'):
                        relative_file = relative_file[2:]
                    if is_dir and relative_file and (relative_file[-1] != '/'):
                        relative_file += '/'
                    loc_list[relative_file] = {'full_name': full_name, 'is_dir': is_dir}
        return (loc_list, single_file)

    def _maintain_cache(cache, local_list):
        if False:
            for i in range(10):
                print('nop')
        if cfg.cache_file and len(cfg.files_from) == 0:
            cache.mark_all_for_purge()
            if PY3:
                local_list_val_iter = local_list.values()
            else:
                local_list_val_iter = local_list.itervalues()
            for f_info in local_list_val_iter:
                inode = f_info.get('inode', 0)
                if not inode:
                    continue
                cache.unmark_for_purge(f_info['dev'], inode, f_info['mtime'], f_info['size'])
            cache.purge()
            cache.save(cfg.cache_file)
    cfg = Config()
    cache = HashCache()
    if cfg.cache_file and os.path.isfile(deunicodise_s(cfg.cache_file)) and (os.path.getsize(deunicodise_s(cfg.cache_file)) > 0):
        cache.load(cfg.cache_file)
    else:
        info(u'Cache file not found or empty, creating/populating it.')
    local_uris = []
    local_list = FileDict(ignore_case=False)
    single_file = False
    if type(args) not in (list, tuple, set):
        args = [args]
    if recursive == None:
        recursive = cfg.recursive
    for arg in args:
        uri = S3Uri(arg)
        if not uri.type == 'file':
            raise ParameterError('Expecting filename or directory instead of: %s' % arg)
        if uri.isdir() and (not recursive):
            raise ParameterError('Use --recursive to upload a directory: %s' % arg)
        local_uris.append(uri)
    for uri in local_uris:
        (list_for_uri, single_file) = _get_filelist_local(local_list, uri, cache, with_dirs)
    if len(local_list) > 1:
        single_file = False
    (local_list, exclude_list) = filter_exclude_include(local_list)
    total_size = _fetch_local_list_info(local_list)
    _maintain_cache(cache, local_list)
    return (local_list, single_file, exclude_list, total_size)

def fetch_remote_list(args, require_attribs=False, recursive=None, uri_params={}):
    if False:
        return 10

    def _get_remote_attribs(uri, remote_item):
        if False:
            i = 10
            return i + 15
        response = S3(cfg).object_info(uri)
        if not response.get('headers'):
            return
        remote_item.update({'size': int(response['headers']['content-length']), 'md5': response['headers']['etag'].strip('"\''), 'timestamp': dateRFC822toUnix(response['headers']['last-modified'])})
        try:
            md5 = response['s3cmd-attrs']['md5']
            remote_item.update({'md5': md5})
            debug(u'retrieved md5=%s from headers' % md5)
        except KeyError:
            pass

    def _get_filelist_remote(remote_uri, recursive=True):
        if False:
            print('Hello World!')
        info(u'Retrieving list of remote files for %s ...' % remote_uri)
        total_size = 0
        s3 = S3(Config())
        response = s3.bucket_list(remote_uri.bucket(), prefix=remote_uri.object(), recursive=recursive, uri_params=uri_params)
        rem_base_original = rem_base = remote_uri.object()
        remote_uri_original = remote_uri
        if rem_base != '' and rem_base[-1] != '/':
            rem_base = rem_base[:rem_base.rfind('/') + 1]
            remote_uri = S3Uri(u's3://%s/%s' % (remote_uri.bucket(), rem_base))
        rem_base_len = len(rem_base)
        rem_list = FileDict(ignore_case=False)
        break_now = False
        for object in response['list']:
            object_key = object['Key']
            object_size = int(object['Size'])
            is_dir = object_key[-1] == '/'
            if object_key == rem_base_original and (not is_dir):
                key = s3path.basename(object_key)
                object_uri_str = remote_uri_original.uri()
                break_now = True
                rem_list = FileDict(ignore_case=False)
            else:
                key = object_key[rem_base_len:]
                object_uri_str = remote_uri.uri() + key
            if not key:
                warning(u'Found empty root object name on S3, ignoring.')
                continue
            rem_list[key] = {'size': object_size, 'timestamp': dateS3toUnix(object['LastModified']), 'md5': object['ETag'].strip('"\''), 'object_key': object_key, 'object_uri_str': object_uri_str, 'base_uri': remote_uri, 'dev': None, 'inode': None, 'is_dir': is_dir}
            if '-' in rem_list[key]['md5']:
                _get_remote_attribs(S3Uri(object_uri_str), rem_list[key])
            md5 = rem_list[key]['md5']
            rem_list.record_md5(key, md5)
            total_size += object_size
            if break_now:
                break
        return (rem_list, total_size)
    cfg = Config()
    remote_uris = []
    remote_list = FileDict(ignore_case=False)
    if type(args) not in (list, tuple, set):
        args = [args]
    if recursive == None:
        recursive = cfg.recursive
    for arg in args:
        uri = S3Uri(arg)
        if not uri.type == 's3':
            raise ParameterError("Expecting S3 URI instead of '%s'" % arg)
        remote_uris.append(uri)
    total_size = 0
    if recursive:
        for uri in remote_uris:
            (objectlist, tmp_total_size) = _get_filelist_remote(uri, recursive=True)
            total_size += tmp_total_size
            for key in objectlist:
                remote_list[key] = objectlist[key]
                remote_list.record_md5(key, objectlist.get_md5(key))
    else:
        for uri in remote_uris:
            uri_str = uri.uri()
            wildcard_split_result = re.split('\\*|\\?', uri_str, maxsplit=1)
            if len(wildcard_split_result) == 2:
                (prefix, rest) = wildcard_split_result
                need_recursion = '/' in rest
                (objectlist, tmp_total_size) = _get_filelist_remote(S3Uri(prefix), recursive=need_recursion)
                total_size += tmp_total_size
                for key in objectlist:
                    if glob.fnmatch.fnmatch(objectlist[key]['object_uri_str'], uri_str):
                        remote_list[key] = objectlist[key]
            else:
                key = s3path.basename(uri.object())
                if not key:
                    raise ParameterError(u'Expecting S3 URI with a filename or --recursive: %s' % uri.uri())
                is_dir = key and key[-1] == '/'
                remote_item = {'base_uri': uri, 'object_uri_str': uri.uri(), 'object_key': uri.object(), 'is_dir': is_dir}
                if require_attribs:
                    _get_remote_attribs(uri, remote_item)
                remote_list[key] = remote_item
                md5 = remote_item.get('md5')
                if md5:
                    remote_list.record_md5(key, md5)
                total_size += remote_item.get('size', 0)
    (remote_list, exclude_list) = filter_exclude_include(remote_list)
    return (remote_list, exclude_list, total_size)

def compare_filelists(src_list, dst_list, src_remote, dst_remote):
    if False:
        while True:
            i = 10

    def __direction_str(is_remote):
        if False:
            print('Hello World!')
        return is_remote and 'remote' or 'local'

    def _compare(src_list, dst_lst, src_remote, dst_remote, file):
        if False:
            for i in range(10):
                print('nop')
        'Return True if src_list[file] matches dst_list[file], else False'
        attribs_match = True
        src_file = src_list.get(file)
        dst_file = dst_list.get(file)
        if not src_file or not dst_file:
            info(u'%s: does not exist in one side or the other: src_list=%s, dst_list=%s' % (file, bool(src_file), bool(dst_file)))
            return False
        if 'size' in cfg.sync_checks:
            src_size = src_file.get('size')
            dst_size = dst_file.get('size')
            if dst_size is not None and src_size is not None and (dst_size != src_size):
                debug(u'xfer: %s (size mismatch: src=%s dst=%s)' % (file, src_size, dst_size))
                attribs_match = False
        compare_md5 = 'md5' in cfg.sync_checks
        if compare_md5:
            if src_remote == True and '-' in src_file['md5'] or (dst_remote == True and '-' in dst_file['md5']):
                compare_md5 = False
                info(u'disabled md5 check for %s' % file)
        if compare_md5 and src_file['is_dir'] == True:
            compare_md5 = False
        if attribs_match and compare_md5:
            try:
                src_md5 = src_list.get_md5(file)
                dst_md5 = dst_list.get_md5(file)
            except (IOError, OSError):
                debug(u'IGNR: %s (disappeared)' % file)
                warning(u'%s: file disappeared, ignoring.' % file)
                raise
            if src_md5 != dst_md5:
                attribs_match = False
                debug(u'XFER: %s (md5 mismatch: src=%s dst=%s)' % (file, src_md5, dst_md5))
        return attribs_match
    assert not (src_remote == False and dst_remote == False)
    info(u'Verifying attributes...')
    cfg = Config()
    update_list = FileDict(ignore_case=False)
    copy_pairs = {}
    debug('Comparing filelists (direction: %s -> %s)' % (__direction_str(src_remote), __direction_str(dst_remote)))
    src_dir_cache = set()
    for relative_file in src_list.keys():
        debug(u"CHECK: '%s'" % relative_file)
        if src_remote:
            dir_idx = relative_file.rfind('/')
            if dir_idx > 0:
                path = relative_file[:dir_idx + 1]
                while path and path not in src_dir_cache:
                    src_dir_cache.add(path)
                    try:
                        path = path[:path.rindex('/', 0, -1) + 1]
                    except ValueError:
                        continue
        if relative_file in dst_list:
            if cfg.skip_existing:
                debug(u"IGNR: '%s' (used --skip-existing)" % relative_file)
                del src_list[relative_file]
                del dst_list[relative_file]
                continue
            try:
                same_file = _compare(src_list, dst_list, src_remote, dst_remote, relative_file)
            except (IOError, OSError):
                debug(u"IGNR: '%s' (disappeared)" % relative_file)
                warning(u'%s: file disappeared, ignoring.' % relative_file)
                del src_list[relative_file]
                del dst_list[relative_file]
                continue
            if same_file:
                debug(u"IGNR: '%s' (transfer not needed)" % relative_file)
                del src_list[relative_file]
                del dst_list[relative_file]
            else:
                try:
                    md5 = src_list.get_md5(relative_file)
                except IOError:
                    md5 = None
                if md5 is not None and md5 in dst_list.by_md5:
                    copy_src_file = dst_list.find_md5_one(md5)
                    debug(u"DST COPY src: '%s' -> '%s'" % (copy_src_file, relative_file))
                    src_item = src_list[relative_file]
                    src_item['md5'] = md5
                    src_item['copy_src'] = copy_src_file
                    copy_pairs[relative_file] = src_item
                    del src_list[relative_file]
                    del dst_list[relative_file]
                else:
                    dst_list.record_md5(relative_file, md5)
                    update_list[relative_file] = src_list[relative_file]
                    del src_list[relative_file]
                    del dst_list[relative_file]
        else:
            try:
                md5 = src_list.get_md5(relative_file)
            except IOError:
                md5 = None
            copy_src_file = dst_list.find_md5_one(md5)
            if copy_src_file is not None:
                debug(u"DST COPY dst: '%s' -> '%s'" % (copy_src_file, relative_file))
                src_item = src_list[relative_file]
                src_item['md5'] = md5
                src_item['copy_src'] = copy_src_file
                copy_pairs[relative_file] = src_item
                del src_list[relative_file]
            else:
                dst_list.record_md5(relative_file, md5)
    for f in dst_list.keys():
        if f in src_list or f in update_list or f in src_dir_cache:
            del dst_list[f]
    return (src_list, dst_list, update_list, copy_pairs)