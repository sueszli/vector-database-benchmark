"""
The cross-reference table contains information that permits random access to indirect objects within the file so
that the entire file need not be read to locate any particular object.
This implementation of PlainTextXREF is invoked whenever the XREF could not be loaded properly.
This implementation will then loop over the entire PDF file to find obj declaration statements.
"""
import io
import logging
import typing
from borb.io.read.tokenize.high_level_tokenizer import HighLevelTokenizer
from borb.io.read.types import Name
from borb.io.read.types import Reference
from borb.pdf.xref.plaintext_xref import PlainTextXREF
from borb.pdf.xref.xref import XREF
logger = logging.getLogger(__name__)

class RebuiltXREF(PlainTextXREF):
    """
    The cross-reference table contains information that permits random access to indirect objects within the file so
    that the entire file need not be read to locate any particular object.
    This implementation of PlainTextXREF is invoked whenever the XREF could not be loaded properly.
    This implementation will then loop over the entire PDF file to find obj declaration statements.
    """

    def read(self, src: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO], tok: HighLevelTokenizer, initial_offset: typing.Optional[int]=None) -> 'XREF':
        if False:
            while True:
                i = 10
        '\n        This method attempts to read a plaintext XREF from the given io_source.\n        It will either throw an exception, or return this XREF\n        '
        logger.info('Switching to RebuiltXREF parsing.')
        pos_before: int = src.tell()
        src.seek(0)
        i: int = 0
        trailer_pos: typing.Optional[int] = None
        bytes_in_pdf: typing.Optional[bytes] = src.read()
        assert bytes_in_pdf is not None, 'rebuilding an XREF is only possible if all the bytes of the PDF are known'
        while i < len(bytes_in_pdf):
            if 48 <= bytes_in_pdf[i] <= 57 and bytes_in_pdf[i + 1] == 32 and (48 <= bytes_in_pdf[i + 2] <= 57) and (bytes_in_pdf[i + 3] == 32) and (bytes_in_pdf[i + 4] == 111) and (bytes_in_pdf[i + 5] == 98) and (bytes_in_pdf[i + 6] == 106):
                logger.debug('%d %d obj at %d' % (bytes_in_pdf[i] - 48, bytes_in_pdf[i + 1] - 48, i))
                self._entries.append(Reference(object_number=bytes_in_pdf[i] - 48, generation_number=bytes_in_pdf[i + 2] - 48, byte_offset=i))
                i += 7
                continue
            if 48 <= bytes_in_pdf[i] <= 57 and 48 <= bytes_in_pdf[i + 1] <= 57 and (bytes_in_pdf[i + 2] == 32) and (48 <= bytes_in_pdf[i + 3] <= 57) and (bytes_in_pdf[i + 4] == 32) and (bytes_in_pdf[i + 5] == 111) and (bytes_in_pdf[i + 6] == 98) and (bytes_in_pdf[i + 7] == 106):
                logger.debug('%d %d obj at %d' % ((bytes_in_pdf[i] - 48) * 10 + (bytes_in_pdf[i + 1] - 48), bytes_in_pdf[i + 3] - 48, i))
                self._entries.append(Reference(object_number=(bytes_in_pdf[i] - 48) * 10 + (bytes_in_pdf[i + 1] - 48), generation_number=bytes_in_pdf[i + 3] - 48, byte_offset=i))
                i += 8
                continue
            if 48 <= bytes_in_pdf[i] <= 57 and 48 <= bytes_in_pdf[i + 1] <= 57 and (48 <= bytes_in_pdf[i + 2] <= 57) and (bytes_in_pdf[i + 3] == 32) and (48 <= bytes_in_pdf[i + 4] <= 57) and (bytes_in_pdf[i + 5] == 32) and (bytes_in_pdf[i + 6] == 111) and (bytes_in_pdf[i + 7] == 98) and (bytes_in_pdf[i + 8] == 106):
                obj_nr: int = (bytes_in_pdf[i] - 48) * 100 + (bytes_in_pdf[i + 1] - 48) * 10 + (bytes_in_pdf[i + 2] - 48)
                logger.debug('%d %d obj at %d' % (obj_nr, bytes_in_pdf[i + 4] - 48, i))
                self._entries.append(Reference(object_number=obj_nr, generation_number=bytes_in_pdf[i + 4] - 48, byte_offset=i))
                i += 9
                continue
            if bytes_in_pdf[i] == 116 and bytes_in_pdf[i + 1] == 114 and (bytes_in_pdf[i + 2] == 97) and (bytes_in_pdf[i + 3] == 105) and (bytes_in_pdf[i + 4] == 108) and (bytes_in_pdf[i + 5] == 101) and (bytes_in_pdf[i + 6] == 114):
                trailer_pos = i
            i += 1
        assert trailer_pos is not None
        src.seek(trailer_pos)
        self[Name('Trailer')] = self._read_trailer(src, tok)
        self[Name('Trailer')].set_parent(self)
        src.seek(pos_before)
        return self