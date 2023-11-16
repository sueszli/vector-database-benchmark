"""
AZLyrics lyrics module.
"""
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from spotdl.providers.lyrics.base import LyricsProvider
__all__ = ['AzLyrics']

class AzLyrics(LyricsProvider):
    """
    AZLyrics lyrics provider class.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.get('https://www.azlyrics.com/')
        resp = self.session.get('https://www.azlyrics.com/geo.js')
        js_code = resp.text
        start_index = js_code.find('value"') + 9
        end_index = js_code[start_index:].find('");')
        self.x_code = js_code[start_index:start_index + end_index]

    def get_results(self, name: str, artists: List[str], **kwargs) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the results for the given song.\n\n        ### Arguments\n        - name: The name of the song.\n        - artists: The artists of the song.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - A dictionary with the results. (The key is the title and the value is the url.)\n        '
        artist_str = ', '.join((artist for artist in artists if artist))
        params = {'q': f'{artist_str} - {name}', 'x': self.x_code}
        counter = 0
        soup = None
        while counter < 4:
            try:
                response = self.session.get('https://search.azlyrics.com/search.php', params=params)
            except requests.ConnectionError:
                continue
            if not response.ok:
                counter += 1
                continue
            soup = BeautifulSoup(response.content, 'html.parser')
            break
        if soup is None:
            return {}
        td_tags = soup.find_all('td')
        if len(td_tags) == 0:
            return {}
        results = {}
        for td_tag in td_tags:
            a_tags = td_tag.find_all('a', href=True)
            if len(a_tags) == 0:
                continue
            a_tag = a_tags[0]
            url = a_tag['href'].strip()
            if url == '':
                continue
            title = td_tag.find('span').get_text().strip()
            artist = td_tag.find('b').get_text().strip()
            results[f'{artist} - {title}'] = url
        return results

    def extract_lyrics(self, url: str, **_) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Extracts the lyrics from the given url.\n\n        ### Arguments\n        - url: The url to extract the lyrics from.\n        - kwargs: Additional arguments.\n\n        ### Returns\n        - The lyrics of the song or None if no lyrics were found.\n        '
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        div_tags = soup.find_all('div', class_=False, id_=False)
        lyrics_div = sorted(div_tags, key=lambda x: len(x.text))[-1]
        lyrics = lyrics_div.get_text().strip()
        return lyrics