"""
MusixMatch lyrics provider.
"""
from typing import Dict, List, Optional
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
from spotdl.providers.lyrics.base import LyricsProvider
__all__ = ['MusixMatch']

class MusixMatch(LyricsProvider):
    """
    MusixMatch lyrics provider class.
    """

    def extract_lyrics(self, url: str, **kwargs) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Extracts the lyrics from the given url.\n\n        ### Arguments\n        - url: The url to extract the lyrics from.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - The lyrics of the song or None if no lyrics were found.\n        '
        lyrics_resp = requests.get(url, headers=self.headers, timeout=10)
        lyrics_soup = BeautifulSoup(lyrics_resp.text, 'html.parser')
        lyrics_paragraphs = lyrics_soup.select('p.mxm-lyrics__content')
        lyrics = '\n'.join((i.get_text() for i in lyrics_paragraphs))
        return lyrics

    def get_results(self, name: str, artists: List[str], **kwargs) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the results for the given song.\n\n        ### Arguments\n        - name: The name of the song.\n        - artists: The artists of the song.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - A dictionary with the results. (The key is the title and the value is the url.)\n        '
        track_search = kwargs.get('track_search', False)
        artists_str = ', '.join((artist for artist in artists if artist.lower() not in name.lower()))
        query = quote(f'{name} - {artists_str}', safe='')
        if track_search:
            query += '/tracks'
        search_url = f'https://www.musixmatch.com/search/{query}'
        search_resp = requests.get(search_url, headers=self.headers, timeout=10)
        search_soup = BeautifulSoup(search_resp.text, 'html.parser')
        song_url_tag = search_soup.select("a[href^='/lyrics/']")
        if not song_url_tag:
            if track_search:
                return {}
            return self.get_results(name, artists, track_search=True)
        results: Dict[str, str] = {}
        for tag in song_url_tag:
            results[tag.get_text()] = 'https://www.musixmatch.com' + str(tag.get('href', ''))
        return results