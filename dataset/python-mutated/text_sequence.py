from typing import Tuple, Union
from aim.sdk.sequence import MediaSequenceBase
from aim.sdk.objects.text import Text

class Texts(MediaSequenceBase):
    """Class representing series of Text objects."""

    @classmethod
    def allowed_dtypes(cls) -> Union[str, Tuple[str, ...]]:
        if False:
            for i in range(10):
                print('nop')
        text_typename = Text.get_typename()
        return (text_typename, f'list({text_typename})')

    @classmethod
    def sequence_name(cls) -> str:
        if False:
            while True:
                i = 10
        return 'texts'