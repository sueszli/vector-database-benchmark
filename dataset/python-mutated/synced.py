"""
Synced lyrics provider using the syncedlyrics library
"""
from typing import Dict, List, Optional
import syncedlyrics
from spotdl.providers.lyrics.base import LyricsProvider
__all__ = ['Synced']

class Synced(LyricsProvider):
    """
    Lyrics provider for synced lyrics using the syncedlyrics library
    Currently supported websites: Deezer, NetEase
    """

    def get_results(self, name: str, artists: List[str], **kwargs) -> Dict[str, str]:
        if False:
            return 10
        '\n        Returns the results for the given song.\n\n        ### Arguments\n        - name: The name of the song.\n        - artists: The artists of the song.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - A dictionary with the results. (The key is the title and the value is the url.)\n        '
        raise NotImplementedError

    def extract_lyrics(self, url: str, **kwargs) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Extracts the lyrics from the given url.\n\n        ### Arguments\n        - url: The url to extract the lyrics from.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - The lyrics of the song or None if no lyrics were found.\n        '
        raise NotImplementedError

    def get_lyrics(self, name: str, artists: List[str], **_) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Try to get lyrics using syncedlyrics\n\n        ### Arguments\n        - name: The name of the song.\n        - artists: The artists of the song.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - The lyrics of the song or None if no lyrics were found.\n        '
        lyrics = syncedlyrics.search(f'{name} - {artists[0]}', allow_plain_format=True)
        return lyrics