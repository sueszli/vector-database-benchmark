import logging
from asyncio import CancelledError, Future
from contextlib import suppress
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional
from tribler.core.components.libtorrent import torrentdef
from tribler.core.components.libtorrent.utils.libtorrent_helper import libtorrent as lt
from tribler.core.utilities.path_util import Path
logger = logging.getLogger(__name__)

def check_handle(default=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the libtorrent handle if it's available, else return the default value.\n    Author(s): Egbert Bouman\n    "

    def wrap(f):
        if False:
            print('Hello World!')

        def invoke_func(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            download = args[0]
            if download.handle and download.handle.is_valid():
                return f(*args, **kwargs)
            return default
        return invoke_func
    return wrap

def require_handle(func):
    if False:
        i = 10
        return i + 15
    '\n    Invoke the function once the handle is available. Returns a future that will fire once the function has completed.\n    Author(s): Egbert Bouman\n    '

    def invoke_func(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result_future = Future()

        def done_cb(fut):
            if False:
                print('Hello World!')
            with suppress(CancelledError):
                handle = fut.result()
            if not fut.cancelled() and (not result_future.done()) and (handle == download.handle) and handle.is_valid() and (not isinstance(download.tdef, torrentdef.TorrentDefNoMetainfo)):
                result_future.set_result(func(*args, **kwargs))
        download = args[0]
        handle_future = download.get_handle()
        handle_future.add_done_callback(done_cb)
        return result_future
    return invoke_func

def check_vod(default=None):
    if False:
        while True:
            i = 10
    '\n    Check if torrent is vod mode, else return default\n    '

    def wrap(f):
        if False:
            print('Hello World!')

        def invoke_func(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if self.enabled:
                return f(self, *args, **kwargs)
            return default
        return invoke_func
    return wrap

def common_prefix(paths_list: List[Path]) -> Path:
    if False:
        i = 10
        return i + 15
    base_set = set(paths_list[0].parents)
    for p in paths_list[1:]:
        base_set.intersection_update(set(p.parents))
    return sorted(base_set, reverse=True)[0]

def _existing_files(path_list: List[Path]) -> Iterable[Path]:
    if False:
        for i in range(10):
            print('nop')
    for path in path_list:
        path = Path(path)
        if not path.exists():
            raise OSError(f'Path does not exist: {path}')
        elif path.is_file():
            yield path

def create_torrent_file(file_path_list: List[Path], params: Dict[bytes, Any], torrent_filepath: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    fs = lt.file_storage()
    path_list = list(_existing_files(file_path_list))
    base_dir = (common_prefix(path_list).parent if len(path_list) > 1 else path_list[0].parent).absolute()
    for path in path_list:
        relative = path.relative_to(base_dir)
        fs.add_file(str(relative), path.size())
    if params.get(b'piece length'):
        piece_size = params[b'piece length']
    else:
        piece_size = 0
    flags = lt.create_torrent_flags_t.optimize
    if hasattr(lt.create_torrent_flags_t, 'calculate_file_hashes'):
        flags |= lt.create_torrent_flags_t.calculate_file_hashes
    params = {k: v.decode('utf-8') if isinstance(v, bytes) else v for (k, v) in params.items()}
    torrent = lt.create_torrent(fs, piece_size=piece_size, flags=flags)
    if params.get(b'comment'):
        torrent.set_comment(params[b'comment'])
    if params.get(b'created by'):
        torrent.set_creator(params[b'created by'])
    if params.get(b'announce'):
        torrent.add_tracker(params[b'announce'])
    if params.get(b'announce-list'):
        tier = 1
        for tracker in params[b'announce-list']:
            torrent.add_tracker(tracker, tier=tier)
            tier += 1
    if params.get(b'nodes'):
        for node in params[b'nodes']:
            torrent.add_node(*node)
    if params.get(b'httpseeds'):
        torrent.add_http_seed(params[b'httpseeds'])
    if len(file_path_list) == 1:
        if params.get(b'urllist', False):
            torrent.add_url_seed(params[b'urllist'])
    lt.set_piece_hashes(torrent, str(base_dir))
    t1 = torrent.generate()
    torrent = lt.bencode(t1)
    if torrent_filepath:
        with open(torrent_filepath, 'wb') as f:
            f.write(torrent)
    return {'success': True, 'base_dir': base_dir, 'torrent_file_path': torrent_filepath, 'metainfo': torrent, 'infohash': sha1(lt.bencode(t1[b'info'])).digest()}

def get_info_from_handle(handle: lt.torrent_handle) -> Optional[lt.torrent_info]:
    if False:
        return 10
    try:
        if hasattr(handle, 'torrent_file'):
            return handle.torrent_file()
        return handle.get_torrent_info()
    except AttributeError as ae:
        logger.warning('No torrent info found from handle: %s', str(ae))
        return None
    except RuntimeError as e:
        logger.warning('Got exception when fetching info from handle: %s', str(e))
        return None