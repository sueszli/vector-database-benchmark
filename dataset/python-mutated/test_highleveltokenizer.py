import io
import typing
import unittest
from borb.io.read.tokenize.high_level_tokenizer import HighLevelTokenizer
from borb.io.read.tokenize.low_level_tokenizer import Token
unittest.TestLoader.sortTestMethodsUsing = None

class TestParseCMAP(unittest.TestCase):

    def test_parse_cmap(self):
        if False:
            print('Hello World!')
        cmap_bytes: bytes = b'\n        <1E> <00660069>\n        <1F> <0066006C>\n        <20> <0020>\n        <2A> <002A>\n        <2C> <002C>\n        <2E> <002E>\n        <3A> <003A>\n        <41> <0041>\n        '
        cmap_tokenizer: HighLevelTokenizer = HighLevelTokenizer(io.BytesIO(cmap_bytes))
        tokens: typing.List[Token] = [cmap_tokenizer.next_non_comment_token() for _ in range(0, 20)]
        assert tokens[0].get_text() == '<1E>'
        assert tokens[1].get_text() == '<00660069>'
        assert tokens[2].get_text() == '<1F>'
        assert tokens[3].get_text() == '<0066006C>'