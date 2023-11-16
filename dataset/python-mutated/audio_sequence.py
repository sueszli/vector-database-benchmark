from typing import Union, Tuple
from aim.sdk.sequence import MediaSequenceBase
from aim.sdk.objects import Audio

class Audios(MediaSequenceBase):
    """Class representing series of Audio objects or Audio lists."""

    @classmethod
    def allowed_dtypes(cls) -> Union[str, Tuple[str, ...]]:
        if False:
            print('Hello World!')
        typename = Audio.get_typename()
        return (typename, f'list({typename})')

    @classmethod
    def sequence_name(cls) -> str:
        if False:
            while True:
                i = 10
        return 'audios'