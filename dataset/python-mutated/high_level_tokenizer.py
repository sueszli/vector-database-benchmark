"""
This module contains everything needed to perform low-level tokenization against PDF syntax.
Low-level tokenization aims to separate numbers, strings, names, comments, start of dictionary, start of array, etc
The high-level tokenizer will use this first pass to then build complex objects (streams, dictionaries, etc)
"""
import re
import typing
from borb.io.read.tokenize.low_level_tokenizer import LowLevelTokenizer
from borb.io.read.tokenize.low_level_tokenizer import TokenType
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Boolean
from borb.io.read.types import CanvasOperatorName
from borb.io.read.types import Decimal as bDecimal
from borb.io.read.types import Dictionary
from borb.io.read.types import HexadecimalString
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import Reference
from borb.io.read.types import Stream
from borb.io.read.types import String

class HighLevelTokenizer(LowLevelTokenizer):
    """
    In computer science, lexical analysis, lexing or tokenization is the process of converting a sequence of characters
    (such as in a computer program or web page) into a sequence of tokens (strings with an assigned and thus identified meaning).
    A program that performs lexical analysis may be termed a lexer, tokenizer, or scanner,
    although scanner is also a term for the first stage of a lexer.
    A lexer is generally combined with a parser, which together analyze the syntax of programming languages, web pages,
    and so forth.
    """

    def read_array(self) -> List:
        if False:
            while True:
                i = 10
        '\n        This function processes the next tokens and returns a List.\n        It fails and throws various errors if the next tokens do not represent a List.\n        '
        token = self.next_non_comment_token()
        assert token is not None
        assert token.get_token_type() == TokenType.START_ARRAY
        out = List()
        while True:
            token = self.next_non_comment_token()
            assert token is not None
            if token.get_token_type() == TokenType.END_ARRAY:
                break
            assert token.get_token_type() != TokenType.END_DICT
            self.seek(token.get_byte_offset())
            obj = self.read_object()
            out.append(obj)
        return out

    def read_dictionary(self) -> Dictionary:
        if False:
            i = 10
            return i + 15
        '\n        This function processes the next tokens and returns a Dictionary.\n        It fails and throws various errors if the next tokens do not represent a Dictionary.\n        '
        token = self.next_non_comment_token()
        assert token is not None
        assert token.get_token_type() == TokenType.START_DICT
        out_dict = Dictionary()
        while True:
            token = self.next_non_comment_token()
            assert token is not None
            if token.get_token_type() == TokenType.END_DICT:
                break
            assert token.get_token_type() == TokenType.NAME
            name = Name(token.get_text()[1:])
            value = self.read_object()
            assert value is not None
            if name is not None:
                out_dict[name] = value
        return out_dict

    def read_indirect_object(self) -> typing.Optional[AnyPDFType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function processes the next tokens and returns an AnyPDFType.\n        It fails and throws various errors if the next tokens do not represent an indirect pdf object.\n        '
        token = self.next_non_comment_token()
        assert token is not None
        byte_offset = token.get_byte_offset()
        if token.get_token_type() != TokenType.NUMBER or not re.match('^[0-9]+$', token.get_text()):
            self.seek(byte_offset)
            return None
        object_number = int(token.get_text())
        token = self.next_non_comment_token()
        assert token is not None
        if token.get_token_type() != TokenType.NUMBER or not re.match('^[0-9]+$', token.get_text()):
            self.seek(byte_offset)
            return None
        generation_number = int(token.get_text())
        token = self.next_non_comment_token()
        assert token is not None
        if token.get_token_type() != TokenType.OTHER or token.get_text() != 'obj':
            self.seek(byte_offset)
            return None
        value = self.read_object()
        if value is not None:
            value.set_reference(Reference(object_number=object_number, generation_number=generation_number))
        return value

    def read_indirect_reference(self) -> typing.Optional[Reference]:
        if False:
            while True:
                i = 10
        '\n        This function processes the next tokens and returns an indirect reference.\n        It fails and throws various errors if the next tokens do not represent an indirect reference.\n        '
        token = self.next_non_comment_token()
        assert token is not None
        byte_offset = token.get_byte_offset()
        if token.get_token_type() != TokenType.NUMBER or not re.match('^[0-9]+$', token.get_text()):
            self.seek(byte_offset)
            return None
        object_number = int(token.get_text())
        token = self.next_non_comment_token()
        assert token is not None
        if token.get_token_type() != TokenType.NUMBER or not re.match('^[0-9]+$', token.get_text()):
            self.seek(byte_offset)
            return None
        generation_number = int(token.get_text())
        token = self.next_non_comment_token()
        assert token is not None
        if token.get_token_type() != TokenType.OTHER or token.get_text() != 'R':
            self.seek(byte_offset)
            return None
        return Reference(object_number=object_number, generation_number=generation_number)

    def read_object(self, xref: typing.Optional['XREF']=None) -> typing.Optional[AnyPDFType]:
        if False:
            return 10
        '\n        This function processes the next tokens and returns an AnyPDFType.\n        It fails and throws various errors if the next tokens do not represent a pdf object.\n        '
        token = self.next_non_comment_token()
        if token is None or len(token.get_text()) == 0:
            return None
        if token.get_token_type() == TokenType.START_DICT:
            self.seek(token.get_byte_offset())
            return self.read_dictionary()
        if token.get_token_type() == TokenType.START_ARRAY:
            self.seek(token.get_byte_offset())
            return self.read_array()
        if token.get_token_type() == TokenType.NUMBER:
            self.seek(token.get_byte_offset())
            potential_indirect_reference = self.read_indirect_reference()
            if potential_indirect_reference is not None:
                return potential_indirect_reference
        if token.get_token_type() == TokenType.NUMBER:
            self.seek(token.get_byte_offset())
            potential_stream = self.read_stream(xref)
            if potential_stream is not None:
                return potential_stream
        if token.get_token_type() == TokenType.NUMBER:
            self.seek(token.get_byte_offset())
            potential_indirect_object = self.read_indirect_object()
            if potential_indirect_object is not None:
                return potential_indirect_object
        if token.get_token_type() == TokenType.NUMBER:
            self.seek(self.tell() + len(token.get_text()))
            return bDecimal(token.get_text())
        if token.get_token_type() == TokenType.OTHER and token.get_text() in ['true', 'false']:
            return Boolean(token.get_text() == 'true')
        if token.get_token_type() == TokenType.OTHER and token.get_text() in CanvasOperatorName.VALID_NAMES:
            return CanvasOperatorName(token.get_text())
        if token.get_token_type() == TokenType.NAME:
            return Name(token.get_text()[1:])
        if token.get_token_type() in [TokenType.STRING, TokenType.HEX_STRING]:
            if token.get_token_type() == TokenType.STRING:
                return String(token.get_text()[1:-1])
            else:
                return HexadecimalString(token.get_text()[1:-1])
        return None

    def read_stream(self, xref: typing.Optional['XREF']=None) -> typing.Optional[Stream]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function processes the next tokens and returns a Stream.\n        It fails and throws various errors if the next tokens do not represent a Stream.\n        '
        byte_offset = self.tell()
        stream_dictionary = self.read_indirect_object()
        if stream_dictionary is None or not isinstance(stream_dictionary, dict):
            self.seek(byte_offset)
            return None
        stream_token = self.next_non_comment_token()
        assert stream_token is not None
        if stream_token.get_token_type() != TokenType.OTHER or stream_token.get_text() != 'stream':
            self.seek(byte_offset)
            return None
        assert 'Length' in stream_dictionary
        length_of_stream = stream_dictionary['Length']
        if isinstance(length_of_stream, Reference):
            if xref is None:
                raise RuntimeError('unable to process reference /Length when no XREF is given')
            pos_before = self.tell()
            length_of_stream = int(xref.get_object(length_of_stream, src=self._io_source, tok=self))
            self.seek(pos_before)
        ch = self._next_byte()
        assert ch in [b'\r', b'\n']
        if ch == b'\r':
            ch = self._next_byte()
            assert ch == b'\n'
        bytes = self._io_source.read(int(length_of_stream))
        end_of_stream_token = self.next_non_comment_token()
        assert end_of_stream_token is not None
        assert end_of_stream_token.get_token_type() == TokenType.OTHER
        assert end_of_stream_token.get_text() == 'endstream'
        stream_dictionary[Name('Bytes')] = bytes
        output: Stream = Stream()
        for (k, v) in stream_dictionary.items():
            output[k] = v
        return output