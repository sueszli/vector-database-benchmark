"""Track python variables in relation to the block graph."""
from typing import Set, Dict, List
from pytype.blocks import blocks
from pytype.typegraph import cfg
LocalValues = List[cfg.Variable]
LocalsDict = Dict[str, LocalValues]
BlockLocals = Dict[blocks.Block, LocalsDict]

class Environment:
    """A store of local variables per blockgraph node."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.block_locals: BlockLocals = {}
        self._dead_ends: Set[blocks.Block] = set()

    def mark_dead_end(self, block):
        if False:
            return 10
        self._dead_ends.add(block)

    def add_block(self, frame, block):
        if False:
            while True:
                i = 10
        'Add a new block and initialize its locals.'
        local = {}
        self.block_locals[block] = local
        incoming = [b for b in block.incoming if b in self.block_locals and b != block and (b not in self._dead_ends)]
        n_inc = len(incoming)
        if n_inc == 0:
            try:
                f_locals = frame.f_locals.pyval.items()
            except AttributeError:
                f_locals = ()
            frame_locals = {k: [v] for (k, v) in f_locals}
            local.update(frame_locals)
        elif n_inc == 1:
            (inc,) = incoming
            local.update(self.block_locals[inc])
        else:
            keys = None
            for b in incoming:
                b_keys = set(self.block_locals[b])
                if keys is None:
                    keys = b_keys
                else:
                    keys &= b_keys
            assert keys is not None
            for k in keys:
                var = set()
                for b in incoming:
                    incoming_locals = self.block_locals[b]
                    var |= set(incoming_locals[k])
                local[k] = list(var)

    def store_local(self, block, name, var):
        if False:
            for i in range(10):
                print('nop')
        self.block_locals[block][name] = [var]

    def get_local(self, block, name):
        if False:
            print('Hello World!')
        return self.block_locals[block].get(name)