from __future__ import print_function
import os
import sys
import codecs
from . import StrParse
from . import pbRoot
from . import pbItem
from .Switch import Switch

def GetFileEncoding(path):
    if False:
        return 10
    encoding = 'utf-8-sig'
    size = os.path.getsize(path)
    if size > 2:
        file_descriptor = OpenFile(path)
        first_two_bytes = file_descriptor.read(2)
        file_descriptor.close()
        for case in Switch(first_two_bytes):
            if case(codecs.BOM_UTF16):
                encoding = 'utf-16'
                break
            if case(codecs.BOM_UTF16_LE):
                encoding = 'utf-16-le'
                break
            if case(codecs.BOM_UTF16_BE):
                encoding = 'utf-16-be'
                break
            if case():
                break
    return encoding

def OpenFileWithEncoding(file_path, encoding):
    if False:
        while True:
            i = 10
    return codecs.open(file_path, 'r', encoding=encoding, errors='ignore')
if sys.version_info < (3, 0):

    def OpenFile(file_path):
        if False:
            return 10
        return open(file_path, 'rb')
else:

    def OpenFile(file_path):
        if False:
            while True:
                i = 10
        return open(file_path, 'br')

class PBParser(object):

    def __init__(self, file_path=None):
        if False:
            print('Hello World!')
        self.index = 0
        self.string_encoding = None
        self.file_path = file_path
        self.file_type = None
        try:
            encoding = GetFileEncoding(self.file_path)
            file_descriptor = OpenFileWithEncoding(self.file_path, encoding)
            self.data = file_descriptor.read()
            if self.file_path.endswith('.strings'):
                self.data = '{' + self.data + '}'
            file_descriptor.close()
        except IOError as exception:
            print('I/O error({0}): {1}'.format(exception.errno, exception.strerror))
        except:
            print('Unexpected error:' + str(sys.exc_info()[0]))
            raise

    def read(self):
        if False:
            i = 10
            return i + 15
        parsed_plist = None
        prefix = self.data[0:6]
        for case in Switch(prefix):
            if case('bplist'):
                break
            if case('<?xml '):
                break
            if case():
                self.file_type = 'ascii'
                if self.data[0:2] == '//':
                    import re
                    result = re.search('^// !\\$\\*(.+?)\\*\\$!', self.data)
                    if result:
                        self.string_encoding = result.group(1)
                parsed_plist = self.__readTest(True)
                break
        return parsed_plist

    def __readTest(self, requires_object=True):
        if False:
            return 10
        read_result = None
        (can_parse, self.index, _annotation) = StrParse.IndexOfNextNonSpace(self.data, self.index)
        if not can_parse:
            if self.index != len(self.data):
                if requires_object is True:
                    message = 'Invalid plist file!'
                    raise Exception(message)
        else:
            read_result = self.__parse(requires_object)
        return read_result

    def __parse(self, requires_object=True):
        if False:
            i = 10
            return i + 15
        parsed_item = None
        starting_character = self.data[self.index]
        for case in Switch(starting_character):
            if case('{'):
                parsed_item = pbItem.pbItemResolver(self.__parseDict(), 'dictionary')
                break
            if case('('):
                parsed_item = pbItem.pbItemResolver(self.__parseArray(), 'array')
                break
            if case('<'):
                parsed_item = pbItem.pbItemResolver(self.__parseData(), 'data')
                break
            if case("'"):
                pass
            if case('"'):
                parsed_item = pbItem.pbItemResolver(self.__parseQuotedString(), 'qstring')
                break
            if case():
                if StrParse.IsValidUnquotedStringCharacter(starting_character) is True:
                    parsed_item = pbItem.pbItemResolver(self.__parseUnquotedString(), 'string')
                elif requires_object is True:
                    message = 'Unexpected character "0x%s" at line %i of file %s' % (str(format(ord(starting_character), 'x')), StrParse.LineNumberForIndex(self.data, self.index), self.file_path)
                    raise Exception(message)
        return parsed_item

    def __parseUnquotedString(self):
        if False:
            while True:
                i = 10
        string_length = len(self.data)
        start_index = self.index
        while self.index < string_length:
            current_char = self.data[self.index]
            if StrParse.IsValidUnquotedStringCharacter(current_char) is True:
                self.index += 1
            else:
                break
        if start_index != self.index:
            return self.data[start_index:self.index]
        else:
            message = 'Unexpected EOF in file %s' % self.file_path
            raise Exception(message)

    def __parseQuotedString(self):
        if False:
            print('Hello World!')
        quote = self.data[self.index]
        string_length = len(self.data)
        self.index += 1
        start_index = self.index
        while self.index < string_length:
            current_char = self.data[self.index]
            if current_char == quote:
                break
            if current_char == '\\':
                self.index += 2
            else:
                self.index += 1
        if self.index >= string_length:
            message = 'Unterminated quoted string starting on line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
            raise Exception(message)
        else:
            string_without_quotes = StrParse.UnQuotifyString(self.data, start_index, self.index)
            self.index += 1
            return string_without_quotes

    def __parseData(self):
        if False:
            for i in range(10):
                print('nop')
        string_length = len(self.data)
        self.index += 1
        start_index = self.index
        end_index = 0
        byte_stream = ''
        while self.index < string_length:
            current_char = self.data[self.index]
            if current_char == '>':
                self.index += 1
                end_index = self.index
                break
            if StrParse.IsHexNumber(current_char) is True:
                byte_stream += current_char
            elif not StrParse.IsDataFormattingWhitespace(current_char):
                message = 'Malformed data byte group (invalid hex) at line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
                raise Exception(message)
            self.index += 1
        if len(byte_stream) % 2 == 1:
            message = 'Malformed data byte group (uneven length) at line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
            raise Exception(message)
        if end_index == 0:
            message = 'Expected terminating >" for data at line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
            raise Exception(message)
        data_object = bytearray.fromhex(byte_stream)
        return data_object

    def __parseArray(self):
        if False:
            for i in range(10):
                print('nop')
        array_objects = list()
        self.index += 1
        start_index = self.index
        new_object = self.__readTest(False)
        while new_object is not None:
            (can_parse, self.index, new_object.annotation) = StrParse.IndexOfNextNonSpace(self.data, self.index)
            _can_parse = can_parse
            array_objects.append(new_object)
            current_char = self.data[self.index]
            if current_char == ',':
                self.index += 1
            new_object = self.__readTest(False)
        current_char = self.data[self.index]
        if current_char != ')':
            message = 'Expected terminating ")" for array at line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
            raise Exception(message)
        self.index += 1
        return array_objects

    def __parseDict(self):
        if False:
            for i in range(10):
                print('nop')
        dictionary = pbRoot.pbRoot()
        self.index += 1
        start_index = self.index
        new_object = self.__readTest(False)
        while new_object is not None:
            (can_parse, self.index, new_object.annotation) = StrParse.IndexOfNextNonSpace(self.data, self.index)
            _can_parse = can_parse
            key_object = new_object
            current_char = self.data[self.index]
            value_object = None
            for case in Switch(current_char):
                if case('='):
                    self.index += 1
                    value_object = self.__readTest(True)
                    break
                if case(';'):
                    self.index += 1
                    value_object = pbItem.pbItemResolver(new_object.value, new_object.type_name)
                    value_object.annotation = new_object.annotation
                    break
                if case():
                    message = 'Missing ";" or "=" on line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
                    raise Exception(message)
            (can_parse, self.index, annotation) = StrParse.IndexOfNextNonSpace(self.data, self.index)
            _can_parse = can_parse
            if value_object.annotation is None:
                value_object.annotation = annotation
            dictionary[key_object] = value_object
            current_char = self.data[self.index]
            if current_char == ';':
                self.index += 1
            new_object = self.__readTest(False)
        current_char = self.data[self.index]
        if current_char != '}':
            message = 'Expected terminating "}" for dictionary at line %s in file %s' % (str(StrParse.LineNumberForIndex(self.data, start_index)), self.file_path)
            raise Exception(message)
        self.index += 1
        return dictionary