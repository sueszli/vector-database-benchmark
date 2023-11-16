from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, List, Set, Tuple
    from manimlib.mobject.mobject import Mobject

def extract_mobject_family_members(mobject_list: Iterable[Mobject], exclude_pointless: bool=False) -> list[Mobject]:
    if False:
        print('Hello World!')
    return [sm for mob in mobject_list for sm in mob.get_family() if not exclude_pointless or sm.has_points()]

def recursive_mobject_remove(mobjects: List[Mobject], to_remove: Set[Mobject]) -> Tuple[List[Mobject], bool]:
    if False:
        i = 10
        return i + 15
    "\n    Takes in a list of mobjects, together with a set of mobjects to remove.\n\n    The first component of what's removed is a new list such that any mobject\n    with one of the elements from `to_remove` in its family is no longer in\n    the list, and in its place are its family members which aren't in `to_remove`\n\n    The second component is a boolean value indicating whether any removals were made\n    "
    result = []
    found_in_list = False
    for mob in mobjects:
        if mob in to_remove:
            found_in_list = True
            continue
        (sub_list, found_in_submobjects) = recursive_mobject_remove(mob.submobjects, to_remove)
        if found_in_submobjects:
            result.extend(sub_list)
            found_in_list = True
        else:
            result.append(mob)
    return (result, found_in_list)