from typing import Union, Tuple
from aim.sdk.sequence import Sequence
from aim.sdk.objects.figure import Figure

class Figures(Sequence):
    """Class representing series of Plotly figure objects or Plotly lists."""

    @classmethod
    def allowed_dtypes(cls) -> Union[str, Tuple[str, ...]]:
        if False:
            print('Hello World!')
        return (Figure.get_typename(),)

    @classmethod
    def sequence_name(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'figures'