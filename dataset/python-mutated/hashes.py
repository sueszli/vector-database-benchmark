import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional
from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks
if TYPE_CHECKING:
    from hashlib import _Hash
    from typing import NoReturn
FAVORITE_HASH = 'sha256'
STRONG_HASHES = ['sha256', 'sha384', 'sha512']

class Hashes:
    """A wrapper that builds multiple hashes at once and checks them against
    known-good values

    """

    def __init__(self, hashes: Optional[Dict[str, List[str]]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param hashes: A dict of algorithm names pointing to lists of allowed\n            hex digests\n        '
        allowed = {}
        if hashes is not None:
            for (alg, keys) in hashes.items():
                allowed[alg] = sorted(keys)
        self._allowed = allowed

    def __and__(self, other: 'Hashes') -> 'Hashes':
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Hashes):
            return NotImplemented
        if not other:
            return self
        if not self:
            return other
        new = {}
        for (alg, values) in other._allowed.items():
            if alg not in self._allowed:
                continue
            new[alg] = [v for v in values if v in self._allowed[alg]]
        return Hashes(new)

    @property
    def digest_count(self) -> int:
        if False:
            while True:
                i = 10
        return sum((len(digests) for digests in self._allowed.values()))

    def is_hash_allowed(self, hash_name: str, hex_digest: str) -> bool:
        if False:
            return 10
        'Return whether the given hex digest is allowed.'
        return hex_digest in self._allowed.get(hash_name, [])

    def check_against_chunks(self, chunks: Iterable[bytes]) -> None:
        if False:
            return 10
        'Check good hashes against ones built from iterable of chunks of\n        data.\n\n        Raise HashMismatch if none match.\n\n        '
        gots = {}
        for hash_name in self._allowed.keys():
            try:
                gots[hash_name] = hashlib.new(hash_name)
            except (ValueError, TypeError):
                raise InstallationError(f'Unknown hash name: {hash_name}')
        for chunk in chunks:
            for hash in gots.values():
                hash.update(chunk)
        for (hash_name, got) in gots.items():
            if got.hexdigest() in self._allowed[hash_name]:
                return
        self._raise(gots)

    def _raise(self, gots: Dict[str, '_Hash']) -> 'NoReturn':
        if False:
            print('Hello World!')
        raise HashMismatch(self._allowed, gots)

    def check_against_file(self, file: BinaryIO) -> None:
        if False:
            print('Hello World!')
        'Check good hashes against a file-like object\n\n        Raise HashMismatch if none match.\n\n        '
        return self.check_against_chunks(read_chunks(file))

    def check_against_path(self, path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        with open(path, 'rb') as file:
            return self.check_against_file(file)

    def has_one_of(self, hashes: Dict[str, str]) -> bool:
        if False:
            print('Hello World!')
        'Return whether any of the given hashes are allowed.'
        for (hash_name, hex_digest) in hashes.items():
            if self.is_hash_allowed(hash_name, hex_digest):
                return True
        return False

    def __bool__(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return whether I know any known-good hashes.'
        return bool(self._allowed)

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Hashes):
            return NotImplemented
        return self._allowed == other._allowed

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash(','.join(sorted((':'.join((alg, digest)) for (alg, digest_list) in self._allowed.items() for digest in digest_list))))

class MissingHashes(Hashes):
    """A workalike for Hashes used when we're missing a hash for a requirement

    It computes the actual hash of the requirement and raises a HashMissing
    exception showing it to the user.

    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        "Don't offer the ``hashes`` kwarg."
        super().__init__(hashes={FAVORITE_HASH: []})

    def _raise(self, gots: Dict[str, '_Hash']) -> 'NoReturn':
        if False:
            while True:
                i = 10
        raise HashMissing(gots[FAVORITE_HASH].hexdigest())