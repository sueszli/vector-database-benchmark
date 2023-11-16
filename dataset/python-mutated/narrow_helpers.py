"""
This module partly exists to prevent circular dependencies.

It also isolates some fairly yucky code related to the fact
that we have to support two formats of narrow specifications
from users:

    legacy:
        [["stream", "devel"], ["is", mentioned"]

    modern:
        [
            {"operator": "stream", "operand": "devel", "negated": "false"},
            {"operator": "is", "operand": "mentioned", "negated": "false"},
        ]

    And then on top of that, we want to represent narrow
    specification internally as dataclasses.
"""
from dataclasses import dataclass
from typing import Collection, Sequence

@dataclass
class NarrowTerm:
    operator: str
    operand: str

def narrow_dataclasses_from_tuples(tups: Collection[Sequence[str]]) -> Collection[NarrowTerm]:
    if False:
        while True:
            i = 10
    '\n    This method assumes that the callers are in our event-handling codepath, and\n    therefore as of summer 2023, they do not yet support the "negated" flag.\n    '
    return [NarrowTerm(operator=tup[0], operand=tup[1]) for tup in tups]