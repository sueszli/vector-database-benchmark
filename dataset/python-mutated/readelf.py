from __future__ import annotations
from enum import Enum
import pwndbg.wrappers
cmd_name = 'readelf'

class RelocationType(Enum):
    JUMP_SLOT = 1
    GLOB_DAT = 2
    IRELATIVE = 3

@pwndbg.wrappers.OnlyWithCommand(cmd_name)
def get_got_entry(local_path: str) -> dict[RelocationType, list[str]]:
    if False:
        for i in range(10):
            print('nop')
    cmd = get_got_entry.cmd + ['--relocs', '--wide', local_path]
    readelf_out = pwndbg.wrappers.call_cmd(cmd)
    entries: dict[RelocationType, list[str]] = {category: [] for category in RelocationType}
    for line in readelf_out.splitlines():
        if not line or not line[0].isdigit() or ' ' not in line:
            continue
        category = line.split()[2]
        for c in RelocationType:
            if c.name in category:
                entries[c].append(line)
    return entries