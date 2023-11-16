from pathlib import Path
from textual.css.stylesheet import Stylesheet

def test_mega_stylesheet() -> None:
    if False:
        return 10
    'It should be possible to load a known-good stylesheet.'
    mega_stylesheet = Stylesheet()
    mega_stylesheet.read(Path(__file__).parent / 'test_mega_stylesheet.tcss')
    mega_stylesheet.parse()
    assert '.---we-made-it-to-the-end---' in mega_stylesheet.css