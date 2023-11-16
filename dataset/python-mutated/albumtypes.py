"""Adds an album template field for formatted album types."""
from beets.autotag.mb import VARIOUS_ARTISTS_ID
from beets.library import Album
from beets.plugins import BeetsPlugin

class AlbumTypesPlugin(BeetsPlugin):
    """Adds an album template field for formatted album types."""

    def __init__(self):
        if False:
            return 10
        'Init AlbumTypesPlugin.'
        super().__init__()
        self.album_template_fields['atypes'] = self._atypes
        self.config.add({'types': [('ep', 'EP'), ('single', 'Single'), ('soundtrack', 'OST'), ('live', 'Live'), ('compilation', 'Anthology'), ('remix', 'Remix')], 'ignore_va': ['compilation'], 'bracket': '[]'})

    def _atypes(self, item: Album):
        if False:
            for i in range(10):
                print('nop')
        "Returns a formatted string based on album's types."
        types = self.config['types'].as_pairs()
        ignore_va = self.config['ignore_va'].as_str_seq()
        bracket = self.config['bracket'].as_str()
        if len(bracket) == 2:
            bracket_l = bracket[0]
            bracket_r = bracket[1]
        else:
            bracket_l = ''
            bracket_r = ''
        res = ''
        albumtypes = item.albumtypes
        is_va = item.mb_albumartistid == VARIOUS_ARTISTS_ID
        for type in types:
            if type[0] in albumtypes and type[1]:
                if not is_va or (type[0] not in ignore_va and is_va):
                    res += f'{bracket_l}{type[1]}{bracket_r}'
        return res