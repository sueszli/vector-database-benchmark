from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AddrSpace:
    """
    Object representing info about the "address space", analogous to the
    LLVM concept. It includes some metadata so that codegen can be
    written in a more generic way.

    Attributes:
        name: human-readable nickname for the address space
        word_scale: a constant which helps calculate offsets in a given
            address space. 1 for word-addressable locations (storage),
            32 for byte-addressable locations (memory, calldata, code)
        load_op: the opcode for loading a word from this address space
        store_op: the opcode for storing a word to this address space
            (an address space is read-only if store_op is None)
    """
    name: str
    word_scale: int
    load_op: str
    store_op: Optional[str] = None

    @property
    def word_addressable(self) -> bool:
        if False:
            while True:
                i = 10
        return self.word_scale == 1

    @property
    def byte_addressable(self) -> bool:
        if False:
            return 10
        return self.word_scale == 32
MEMORY = AddrSpace('memory', 32, 'mload', 'mstore')
STORAGE = AddrSpace('storage', 1, 'sload', 'sstore')
TRANSIENT = AddrSpace('transient', 1, 'tload', 'tstore')
CALLDATA = AddrSpace('calldata', 32, 'calldataload')
IMMUTABLES = AddrSpace('immutables', 32, 'iload', 'istore')
DATA = AddrSpace('data', 32, 'dload')