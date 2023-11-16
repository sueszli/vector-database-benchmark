"""
Compatibility layer with Python 3.8/3.9
"""
from typing import TYPE_CHECKING, Any, Optional
if TYPE_CHECKING:
    from . import Distribution, EntryPoint
else:
    Distribution = EntryPoint = Any

def normalized_name(dist: Distribution) -> Optional[str]:
    if False:
        print('Hello World!')
    "\n    Honor name normalization for distributions that don't provide ``_normalized_name``.\n    "
    try:
        return dist._normalized_name
    except AttributeError:
        from . import Prepared
        return Prepared.normalize(getattr(dist, 'name', None) or dist.metadata['Name'])

def ep_matches(ep: EntryPoint, **params) -> bool:
    if False:
        return 10
    '\n    Workaround for ``EntryPoint`` objects without the ``matches`` method.\n    '
    try:
        return ep.matches(**params)
    except AttributeError:
        from . import EntryPoint
        return EntryPoint(ep.name, ep.value, ep.group).matches(**params)