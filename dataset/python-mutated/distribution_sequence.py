from typing import Tuple
from aim.sdk.sequence import Sequence
from aim.sdk.objects.distribution import Distribution

class Distributions(Sequence):
    """Class representing series of Distribution objects."""

    @classmethod
    def allowed_dtypes(cls) -> Tuple[str, ...]:
        if False:
            return 10
        return (Distribution.get_typename(),)

    @classmethod
    def sequence_name(cls) -> str:
        if False:
            i = 10
            return i + 15
        return 'distributions'