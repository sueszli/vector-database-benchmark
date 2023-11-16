import os
import threading
import time
from typing import Optional, Dict, Mapping, Sequence, TYPE_CHECKING
from . import util
from .bitcoin import hash_encode, int_to_hex, rev_hex
from .crypto import sha256d
from . import constants
from .util import bfh, with_lock
from .logging import get_logger, Logger
if TYPE_CHECKING:
    from .simple_config import SimpleConfig
_logger = get_logger(__name__)
HEADER_SIZE = 80
MAX_TARGET = 26959946667150639794667015087019630673637144422540572481103610249215

class MissingHeader(Exception):
    pass

class InvalidHeader(Exception):
    pass

def serialize_header(header_dict: dict) -> str:
    if False:
        print('Hello World!')
    s = int_to_hex(header_dict['version'], 4) + rev_hex(header_dict['prev_block_hash']) + rev_hex(header_dict['merkle_root']) + int_to_hex(int(header_dict['timestamp']), 4) + int_to_hex(int(header_dict['bits']), 4) + int_to_hex(int(header_dict['nonce']), 4)
    return s

def deserialize_header(s: bytes, height: int) -> dict:
    if False:
        while True:
            i = 10
    if not s:
        raise InvalidHeader('Invalid header: {}'.format(s))
    if len(s) != HEADER_SIZE:
        raise InvalidHeader('Invalid header length: {}'.format(len(s)))
    hex_to_int = lambda s: int.from_bytes(s, byteorder='little')
    h = {}
    h['version'] = hex_to_int(s[0:4])
    h['prev_block_hash'] = hash_encode(s[4:36])
    h['merkle_root'] = hash_encode(s[36:68])
    h['timestamp'] = hex_to_int(s[68:72])
    h['bits'] = hex_to_int(s[72:76])
    h['nonce'] = hex_to_int(s[76:80])
    h['block_height'] = height
    return h

def hash_header(header: dict) -> str:
    if False:
        i = 10
        return i + 15
    if header is None:
        return '0' * 64
    if header.get('prev_block_hash') is None:
        header['prev_block_hash'] = '00' * 32
    return hash_raw_header(serialize_header(header))

def hash_raw_header(header: str) -> str:
    if False:
        i = 10
        return i + 15
    return hash_encode(sha256d(bfh(header)))
pow_hash_header = hash_header
blockchains = {}
blockchains_lock = threading.RLock()

def read_blockchains(config: 'SimpleConfig'):
    if False:
        return 10
    best_chain = Blockchain(config=config, forkpoint=0, parent=None, forkpoint_hash=constants.net.GENESIS, prev_hash=None)
    blockchains[constants.net.GENESIS] = best_chain
    if best_chain.height() > constants.net.max_checkpoint():
        header_after_cp = best_chain.read_header(constants.net.max_checkpoint() + 1)
        if not header_after_cp or not best_chain.can_connect(header_after_cp, check_height=False):
            _logger.info('[blockchain] deleting best chain. cannot connect header after last cp to last cp.')
            os.unlink(best_chain.path())
            best_chain.update_size()
    fdir = os.path.join(util.get_headers_dir(config), 'forks')
    util.make_dir(fdir)
    l = filter(lambda x: x.startswith('fork2_') and '.' not in x, os.listdir(fdir))
    l = sorted(l, key=lambda x: int(x.split('_')[1]))

    def delete_chain(filename, reason):
        if False:
            for i in range(10):
                print('nop')
        _logger.info(f'[blockchain] deleting chain {filename}: {reason}')
        os.unlink(os.path.join(fdir, filename))

    def instantiate_chain(filename):
        if False:
            for i in range(10):
                print('nop')
        (__, forkpoint, prev_hash, first_hash) = filename.split('_')
        forkpoint = int(forkpoint)
        prev_hash = (64 - len(prev_hash)) * '0' + prev_hash
        first_hash = (64 - len(first_hash)) * '0' + first_hash
        if forkpoint <= constants.net.max_checkpoint():
            delete_chain(filename, 'deleting fork below max checkpoint')
            return
        for parent in blockchains.values():
            if parent.check_hash(forkpoint - 1, prev_hash):
                break
        else:
            delete_chain(filename, 'cannot find parent for chain')
            return
        b = Blockchain(config=config, forkpoint=forkpoint, parent=parent, forkpoint_hash=first_hash, prev_hash=prev_hash)
        h = b.read_header(b.forkpoint)
        if first_hash != hash_header(h):
            delete_chain(filename, 'incorrect first hash for chain')
            return
        if not b.parent.can_connect(h, check_height=False):
            delete_chain(filename, 'cannot connect chain to parent')
            return
        chain_id = b.get_id()
        assert first_hash == chain_id, (first_hash, chain_id)
        blockchains[chain_id] = b
    for filename in l:
        instantiate_chain(filename)

def get_best_chain() -> 'Blockchain':
    if False:
        return 10
    return blockchains[constants.net.GENESIS]
_CHAINWORK_CACHE = {'0000000000000000000000000000000000000000000000000000000000000000': 0}

def init_headers_file_for_best_chain():
    if False:
        for i in range(10):
            print('nop')
    b = get_best_chain()
    filename = b.path()
    length = HEADER_SIZE * len(constants.net.CHECKPOINTS) * 2016
    if not os.path.exists(filename) or os.path.getsize(filename) < length:
        with open(filename, 'wb') as f:
            if length > 0:
                f.seek(length - 1)
                f.write(b'\x00')
        util.ensure_sparse_file(filename)
    with b.lock:
        b.update_size()

class Blockchain(Logger):
    """
    Manages blockchain headers and their verification
    """

    def __init__(self, config: 'SimpleConfig', forkpoint: int, parent: Optional['Blockchain'], forkpoint_hash: str, prev_hash: Optional[str]):
        if False:
            return 10
        assert isinstance(forkpoint_hash, str) and len(forkpoint_hash) == 64, forkpoint_hash
        assert prev_hash is None or (isinstance(prev_hash, str) and len(prev_hash) == 64), prev_hash
        if 0 < forkpoint <= constants.net.max_checkpoint():
            raise Exception(f'cannot fork below max checkpoint. forkpoint: {forkpoint}')
        Logger.__init__(self)
        self.config = config
        self.forkpoint = forkpoint
        self.parent = parent
        self._forkpoint_hash = forkpoint_hash
        self._prev_hash = prev_hash
        self.lock = threading.RLock()
        self.update_size()

    @property
    def checkpoints(self):
        if False:
            for i in range(10):
                print('nop')
        return constants.net.CHECKPOINTS

    def get_max_child(self) -> Optional[int]:
        if False:
            return 10
        children = self.get_direct_children()
        return max([x.forkpoint for x in children]) if children else None

    def get_max_forkpoint(self) -> int:
        if False:
            print('Hello World!')
        'Returns the max height where there is a fork\n        related to this chain.\n        '
        mc = self.get_max_child()
        return mc if mc is not None else self.forkpoint

    def get_direct_children(self) -> Sequence['Blockchain']:
        if False:
            while True:
                i = 10
        with blockchains_lock:
            return list(filter(lambda y: y.parent == self, blockchains.values()))

    def get_parent_heights(self) -> Mapping['Blockchain', int]:
        if False:
            print('Hello World!')
        'Returns map: (parent chain -> height of last common block)'
        with self.lock, blockchains_lock:
            result = {self: self.height()}
            chain = self
            while True:
                parent = chain.parent
                if parent is None:
                    break
                result[parent] = chain.forkpoint - 1
                chain = parent
            return result

    def get_height_of_last_common_block_with_chain(self, other_chain: 'Blockchain') -> int:
        if False:
            return 10
        last_common_block_height = 0
        our_parents = self.get_parent_heights()
        their_parents = other_chain.get_parent_heights()
        for chain in our_parents:
            if chain in their_parents:
                h = min(our_parents[chain], their_parents[chain])
                last_common_block_height = max(last_common_block_height, h)
        return last_common_block_height

    @with_lock
    def get_branch_size(self) -> int:
        if False:
            return 10
        return self.height() - self.get_max_forkpoint() + 1

    def get_name(self) -> str:
        if False:
            return 10
        return self.get_hash(self.get_max_forkpoint()).lstrip('0')[0:10]

    def check_header(self, header: dict) -> bool:
        if False:
            i = 10
            return i + 15
        header_hash = hash_header(header)
        height = header.get('block_height')
        return self.check_hash(height, header_hash)

    def check_hash(self, height: int, header_hash: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns whether the hash of the block at given height\n        is the given hash.\n        '
        assert isinstance(header_hash, str) and len(header_hash) == 64, header_hash
        try:
            return header_hash == self.get_hash(height)
        except Exception:
            return False

    def fork(parent, header: dict) -> 'Blockchain':
        if False:
            for i in range(10):
                print('nop')
        if not parent.can_connect(header, check_height=False):
            raise Exception('forking header does not connect to parent chain')
        forkpoint = header.get('block_height')
        self = Blockchain(config=parent.config, forkpoint=forkpoint, parent=parent, forkpoint_hash=hash_header(header), prev_hash=parent.get_hash(forkpoint - 1))
        self.assert_headers_file_available(parent.path())
        open(self.path(), 'w+').close()
        self.save_header(header)
        chain_id = self.get_id()
        with blockchains_lock:
            blockchains[chain_id] = self
        return self

    @with_lock
    def height(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.forkpoint + self.size() - 1

    @with_lock
    def size(self) -> int:
        if False:
            while True:
                i = 10
        return self._size

    @with_lock
    def update_size(self) -> None:
        if False:
            i = 10
            return i + 15
        p = self.path()
        self._size = os.path.getsize(p) // HEADER_SIZE if os.path.exists(p) else 0

    @classmethod
    def verify_header(cls, header: dict, prev_hash: str, target: int, expected_header_hash: str=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        _hash = hash_header(header)
        if expected_header_hash and expected_header_hash != _hash:
            raise InvalidHeader('hash mismatches with expected: {} vs {}'.format(expected_header_hash, _hash))
        if prev_hash != header.get('prev_block_hash'):
            raise InvalidHeader('prev hash mismatch: %s vs %s' % (prev_hash, header.get('prev_block_hash')))
        if constants.net.TESTNET:
            return
        bits = cls.target_to_bits(target)
        if bits != header.get('bits'):
            raise InvalidHeader('bits mismatch: %s vs %s' % (bits, header.get('bits')))
        _pow_hash = pow_hash_header(header)
        pow_hash_as_num = int.from_bytes(bfh(_pow_hash), byteorder='big')
        if pow_hash_as_num > target:
            raise InvalidHeader(f'insufficient proof of work: {pow_hash_as_num} vs target {target}')

    def verify_chunk(self, index: int, data: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        num = len(data) // HEADER_SIZE
        start_height = index * 2016
        prev_hash = self.get_hash(start_height - 1)
        target = self.get_target(index - 1)
        for i in range(num):
            height = start_height + i
            try:
                expected_header_hash = self.get_hash(height)
            except MissingHeader:
                expected_header_hash = None
            raw_header = data[i * HEADER_SIZE:(i + 1) * HEADER_SIZE]
            header = deserialize_header(raw_header, index * 2016 + i)
            self.verify_header(header, prev_hash, target, expected_header_hash)
            prev_hash = hash_header(header)

    @with_lock
    def path(self):
        if False:
            return 10
        d = util.get_headers_dir(self.config)
        if self.parent is None:
            filename = 'blockchain_headers'
        else:
            assert self.forkpoint > 0, self.forkpoint
            prev_hash = self._prev_hash.lstrip('0')
            first_hash = self._forkpoint_hash.lstrip('0')
            basename = f'fork2_{self.forkpoint}_{prev_hash}_{first_hash}'
            filename = os.path.join('forks', basename)
        return os.path.join(d, filename)

    @with_lock
    def save_chunk(self, index: int, chunk: bytes):
        if False:
            i = 10
            return i + 15
        assert index >= 0, index
        chunk_within_checkpoint_region = index < len(self.checkpoints)
        if chunk_within_checkpoint_region and self.parent is not None:
            main_chain = get_best_chain()
            main_chain.save_chunk(index, chunk)
            return
        delta_height = index * 2016 - self.forkpoint
        delta_bytes = delta_height * HEADER_SIZE
        if delta_bytes < 0:
            chunk = chunk[-delta_bytes:]
            delta_bytes = 0
        truncate = not chunk_within_checkpoint_region
        self.write(chunk, delta_bytes, truncate)
        self.swap_with_parent()

    def swap_with_parent(self) -> None:
        if False:
            return 10
        with self.lock, blockchains_lock:
            cnt = 0
            while True:
                old_parent = self.parent
                if not self._swap_with_parent():
                    break
                cnt += 1
                if cnt > len(blockchains):
                    raise Exception(f'swapping fork with parent too many times: {cnt}')
                for old_sibling in old_parent.get_direct_children():
                    if self.check_hash(old_sibling.forkpoint - 1, old_sibling._prev_hash):
                        old_sibling.parent = self

    def _swap_with_parent(self) -> bool:
        if False:
            i = 10
            return i + 15
        "Check if this chain became stronger than its parent, and swap\n        the underlying files if so. The Blockchain instances will keep\n        'containing' the same headers, but their ids change and so\n        they will be stored in different files."
        if self.parent is None:
            return False
        if self.parent.get_chainwork() >= self.get_chainwork():
            return False
        self.logger.info(f'swapping {self.forkpoint} {self.parent.forkpoint}')
        parent_branch_size = self.parent.height() - self.forkpoint + 1
        forkpoint = self.forkpoint
        parent = self.parent
        child_old_id = self.get_id()
        parent_old_id = parent.get_id()
        self.assert_headers_file_available(self.path())
        child_old_name = self.path()
        with open(self.path(), 'rb') as f:
            my_data = f.read()
        self.assert_headers_file_available(parent.path())
        assert forkpoint > parent.forkpoint, f"forkpoint of parent chain ({parent.forkpoint}) should be at lower height than children's ({forkpoint})"
        with open(parent.path(), 'rb') as f:
            f.seek((forkpoint - parent.forkpoint) * HEADER_SIZE)
            parent_data = f.read(parent_branch_size * HEADER_SIZE)
        self.write(parent_data, 0)
        parent.write(my_data, (forkpoint - parent.forkpoint) * HEADER_SIZE)
        (self.parent, parent.parent) = (parent.parent, self)
        (self.forkpoint, parent.forkpoint) = (parent.forkpoint, self.forkpoint)
        (self._forkpoint_hash, parent._forkpoint_hash) = (parent._forkpoint_hash, hash_raw_header(parent_data[:HEADER_SIZE].hex()))
        (self._prev_hash, parent._prev_hash) = (parent._prev_hash, self._prev_hash)
        os.replace(child_old_name, parent.path())
        self.update_size()
        parent.update_size()
        blockchains.pop(child_old_id, None)
        blockchains.pop(parent_old_id, None)
        blockchains[self.get_id()] = self
        blockchains[parent.get_id()] = parent
        return True

    def get_id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._forkpoint_hash

    def assert_headers_file_available(self, path):
        if False:
            i = 10
            return i + 15
        if os.path.exists(path):
            return
        elif not os.path.exists(util.get_headers_dir(self.config)):
            raise FileNotFoundError('Electrum headers_dir does not exist. Was it deleted while running?')
        else:
            raise FileNotFoundError('Cannot find headers file but headers_dir is there. Should be at {}'.format(path))

    @with_lock
    def write(self, data: bytes, offset: int, truncate: bool=True) -> None:
        if False:
            print('Hello World!')
        filename = self.path()
        self.assert_headers_file_available(filename)
        with open(filename, 'rb+') as f:
            if truncate and offset != self._size * HEADER_SIZE:
                f.seek(offset)
                f.truncate()
            f.seek(offset)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        self.update_size()

    @with_lock
    def save_header(self, header: dict) -> None:
        if False:
            i = 10
            return i + 15
        delta = header.get('block_height') - self.forkpoint
        data = bfh(serialize_header(header))
        assert delta == self.size(), (delta, self.size())
        assert len(data) == HEADER_SIZE
        self.write(data, delta * HEADER_SIZE)
        self.swap_with_parent()

    @with_lock
    def read_header(self, height: int) -> Optional[dict]:
        if False:
            i = 10
            return i + 15
        if height < 0:
            return
        if height < self.forkpoint:
            return self.parent.read_header(height)
        if height > self.height():
            return
        delta = height - self.forkpoint
        name = self.path()
        self.assert_headers_file_available(name)
        with open(name, 'rb') as f:
            f.seek(delta * HEADER_SIZE)
            h = f.read(HEADER_SIZE)
            if len(h) < HEADER_SIZE:
                raise Exception('Expected to read a full header. This was only {} bytes'.format(len(h)))
        if h == bytes([0]) * HEADER_SIZE:
            return None
        return deserialize_header(h, height)

    def header_at_tip(self) -> Optional[dict]:
        if False:
            return 10
        'Return latest header.'
        height = self.height()
        return self.read_header(height)

    def is_tip_stale(self) -> bool:
        if False:
            i = 10
            return i + 15
        STALE_DELAY = 8 * 60 * 60
        header = self.header_at_tip()
        if not header:
            return True
        if header['timestamp'] + STALE_DELAY < time.time():
            return True
        return False

    def get_hash(self, height: int) -> str:
        if False:
            for i in range(10):
                print('nop')

        def is_height_checkpoint():
            if False:
                while True:
                    i = 10
            within_cp_range = height <= constants.net.max_checkpoint()
            at_chunk_boundary = (height + 1) % 2016 == 0
            return within_cp_range and at_chunk_boundary
        if height == -1:
            return '0000000000000000000000000000000000000000000000000000000000000000'
        elif height == 0:
            return constants.net.GENESIS
        elif is_height_checkpoint():
            index = height // 2016
            (h, t) = self.checkpoints[index]
            return h
        else:
            header = self.read_header(height)
            if header is None:
                raise MissingHeader(height)
            return hash_header(header)

    def get_target(self, index: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        if constants.net.TESTNET:
            return 0
        if index == -1:
            return MAX_TARGET
        if index < len(self.checkpoints):
            (h, t) = self.checkpoints[index]
            return t
        first = self.read_header(index * 2016)
        last = self.read_header(index * 2016 + 2015)
        if not first or not last:
            raise MissingHeader()
        bits = last.get('bits')
        target = self.bits_to_target(bits)
        nActualTimespan = last.get('timestamp') - first.get('timestamp')
        nTargetTimespan = 14 * 24 * 60 * 60
        nActualTimespan = max(nActualTimespan, nTargetTimespan // 4)
        nActualTimespan = min(nActualTimespan, nTargetTimespan * 4)
        new_target = min(MAX_TARGET, target * nActualTimespan // nTargetTimespan)
        new_target = self.bits_to_target(self.target_to_bits(new_target))
        return new_target

    @classmethod
    def bits_to_target(cls, bits: int) -> int:
        if False:
            print('Hello World!')
        if not 0 <= bits < 1 << 32:
            raise InvalidHeader(f'bits should be uint32. got {bits!r}')
        bitsN = bits >> 24 & 255
        bitsBase = bits & 8388607
        if bitsN <= 3:
            target = bitsBase >> 8 * (3 - bitsN)
        else:
            target = bitsBase << 8 * (bitsN - 3)
        if target != 0 and bits & 8388608 != 0:
            raise InvalidHeader('target cannot be negative')
        if target != 0 and (bitsN > 34 or (bitsN > 33 and bitsBase > 255) or (bitsN > 32 and bitsBase > 65535)):
            raise InvalidHeader('target has overflown')
        return target

    @classmethod
    def target_to_bits(cls, target: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        c = target.to_bytes(length=32, byteorder='big')
        bitsN = len(c)
        while bitsN > 0 and c[0] == 0:
            c = c[1:]
            bitsN -= 1
            if len(c) < 3:
                c += b'\x00'
        bitsBase = int.from_bytes(c[:3], byteorder='big')
        if bitsBase >= 8388608:
            bitsN += 1
            bitsBase >>= 8
        return bitsN << 24 | bitsBase

    def chainwork_of_header_at_height(self, height: int) -> int:
        if False:
            print('Hello World!')
        'work done by single header at given height'
        chunk_idx = height // 2016 - 1
        target = self.get_target(chunk_idx)
        work = (2 ** 256 - target - 1) // (target + 1) + 1
        return work

    @with_lock
    def get_chainwork(self, height=None) -> int:
        if False:
            print('Hello World!')
        if height is None:
            height = max(0, self.height())
        if constants.net.TESTNET:
            return height
        last_retarget = height // 2016 * 2016 - 1
        cached_height = last_retarget
        while _CHAINWORK_CACHE.get(self.get_hash(cached_height)) is None:
            if cached_height <= -1:
                break
            cached_height -= 2016
        assert cached_height >= -1, cached_height
        running_total = _CHAINWORK_CACHE[self.get_hash(cached_height)]
        while cached_height < last_retarget:
            cached_height += 2016
            work_in_single_header = self.chainwork_of_header_at_height(cached_height)
            work_in_chunk = 2016 * work_in_single_header
            running_total += work_in_chunk
            _CHAINWORK_CACHE[self.get_hash(cached_height)] = running_total
        cached_height += 2016
        work_in_single_header = self.chainwork_of_header_at_height(cached_height)
        work_in_last_partial_chunk = (height % 2016 + 1) * work_in_single_header
        return running_total + work_in_last_partial_chunk

    def can_connect(self, header: dict, check_height: bool=True) -> bool:
        if False:
            return 10
        if header is None:
            return False
        height = header['block_height']
        if check_height and self.height() != height - 1:
            return False
        if height == 0:
            return hash_header(header) == constants.net.GENESIS
        try:
            prev_hash = self.get_hash(height - 1)
        except Exception:
            return False
        if prev_hash != header.get('prev_block_hash'):
            return False
        try:
            target = self.get_target(height // 2016 - 1)
        except MissingHeader:
            return False
        try:
            self.verify_header(header, prev_hash, target)
        except BaseException as e:
            return False
        return True

    def connect_chunk(self, idx: int, hexdata: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        assert idx >= 0, idx
        try:
            data = bfh(hexdata)
            self.verify_chunk(idx, data)
            self.save_chunk(idx, data)
            return True
        except BaseException as e:
            self.logger.info(f'verify_chunk idx {idx} failed: {repr(e)}')
            return False

    def get_checkpoints(self):
        if False:
            for i in range(10):
                print('nop')
        cp = []
        n = self.height() // 2016
        for index in range(n):
            h = self.get_hash((index + 1) * 2016 - 1)
            target = self.get_target(index)
            cp.append((h, target))
        return cp

def check_header(header: dict) -> Optional[Blockchain]:
    if False:
        while True:
            i = 10
    'Returns any Blockchain that contains header, or None.'
    if type(header) is not dict:
        return None
    with blockchains_lock:
        chains = list(blockchains.values())
    for b in chains:
        if b.check_header(header):
            return b
    return None

def can_connect(header: dict) -> Optional[Blockchain]:
    if False:
        print('Hello World!')
    'Returns the Blockchain that has a tip that directly links up\n    with header, or None.\n    '
    with blockchains_lock:
        chains = list(blockchains.values())
    for b in chains:
        if b.can_connect(header):
            return b
    return None

def get_chains_that_contain_header(height: int, header_hash: str) -> Sequence[Blockchain]:
    if False:
        return 10
    'Returns a list of Blockchains that contain header, best chain first.'
    with blockchains_lock:
        chains = list(blockchains.values())
    chains = [chain for chain in chains if chain.check_hash(height=height, header_hash=header_hash)]
    chains = sorted(chains, key=lambda x: x.get_chainwork(), reverse=True)
    return chains