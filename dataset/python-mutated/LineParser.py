import logging
import re
from coala_utils.string_processing.StringConverter import StringConverter
from coala_utils.string_processing import unescape, convert_to_raw, position_is_escaped, unescaped_rstrip

class LineParser:

    def __init__(self, key_value_delimiters=('=',), comment_separators=('#',), key_delimiters=(',', ' '), section_name_surroundings=None, section_override_delimiters=('.',), key_value_append_delimiters=('+=',)):
        if False:
            return 10
        '\n        Creates a new line parser. Please note that no delimiter or separator\n        may be an "o" or you may encounter undefined behaviour with the\n        escapes.\n\n        :param key_value_delimiters:        Delimiters that delimit a key from\n                                            a value.\n        :param comment_separators:          Used to initiate a comment.\n        :param key_delimiters:              Delimiters between several keys.\n        :param section_name_surroundings:   Dictionary, e.g. {"[", "]"} means a\n                                            section name is surrounded by [].\n                                            If None, {"[": "]"} is used as\n                                            default.\n        :param section_override_delimiters: Delimiter for a section override.\n                                            E.g. "." would mean that\n                                            section.key is a possible key that\n                                            puts the key into the section\n                                            "section" despite of the current\n                                            section.\n        :param key_value_append_delimiters: Delimiters to separate key and\n                                            value in setting arguments where\n                                            settings are being appended.\n        '
        section_name_surroundings = {'[': ']'} if section_name_surroundings is None else section_name_surroundings
        self.key_value_delimiters = key_value_delimiters
        self.key_value_append_delimiters = key_value_append_delimiters
        self.comment_separators = comment_separators
        self.key_delimiters = key_delimiters
        self.section_name_surroundings = section_name_surroundings
        self.section_override_delimiters = section_override_delimiters

    def parse(self, line):
        if False:
            i = 10
            return i + 15
        "\n        Note that every value in the returned tuple *besides the value* is\n        unescaped. This is so since the value is meant to be put into a Setting\n        later thus the escapes may be needed there.\n\n        :param line: The line to parse.\n        :return:     section_name (empty string if it's no section name),\n                     [(section_override, key), ...], value, comment\n        "
        logging.warning('The parse method of LineParser is deprecated and will be removed. Please use `_parse` which has a new return type, a tuple containing 5 values instead of 4. Refer to the method documentation for further information.')
        (section_name, key_tuples, value, _, comment) = self._parse(line)
        return (section_name, key_tuples, value, comment)

    def _parse(self, line):
        if False:
            return 10
        "\n        Note that every value in the returned tuple *besides the value* is\n        unescaped. This is so since the value is meant to be put into a Setting\n        later thus the escapes may be needed there.\n\n        :param line: The line to parse.\n        :return:     section_name (empty string if it's no section name),\n                     [(section_override, key), ...], value, to_append (True if\n                     append delimiter is found else False), comment\n        "
        for separator in self.comment_separators:
            if re.match('[^ ]' + separator, line) or re.match(separator + '[^ ]', line):
                logging.warning('This comment does not have whitespace' + ' before or after ' + separator + ' in: ' + repr(line.replace('\n', '')) + '. If you ' + "didn't mean to make a comment, use a " + 'backslash for escaping.')
        (line, comment) = self.__separate_by_first_occurrence(line, self.comment_separators)
        comment = unescape(comment)
        if line == '':
            return ('', [], '', False, comment)
        section_name = unescape(self.__get_section_name(line))
        if section_name != '':
            return (section_name, [], '', False, comment)
        append = True
        (keys, value) = self.__extract_keys_and_value(line, self.key_value_append_delimiters)
        if not value:
            (keys, value) = self.__extract_keys_and_value(line, self.key_value_delimiters, True)
            append = False
        all_delimiters = self.key_value_delimiters
        all_delimiters += self.key_value_append_delimiters
        all_delimiters += self.key_delimiters
        all_delimiters += self.comment_separators
        all_delimiters += self.section_override_delimiters
        all_delimiters = ''.join(all_delimiters)
        all_delimiters += ''.join(self.section_name_surroundings.keys())
        all_delimiters += ''.join(self.section_name_surroundings.values())
        value = convert_to_raw(value, all_delimiters)
        key_tuples = []
        for key in keys:
            key = convert_to_raw(key, all_delimiters)
            (section, key) = self.__separate_by_first_occurrence(key, self.section_override_delimiters, True, True)
            key_tuples.append((unescape(section), unescape(key)))
        return ('', key_tuples, value, append, comment)

    @staticmethod
    def __separate_by_first_occurrence(string, delimiters, strip_delim=False, return_second_part_nonempty=False):
        if False:
            while True:
                i = 10
        '\n        Separates a string by the first of all given delimiters. Any whitespace\n        characters will be stripped away from the parts.\n\n        :param string:                      The string to separate.\n        :param delimiters:                  The delimiters.\n        :param strip_delim:                 Strips the delimiter from the\n                                            result if true.\n        :param return_second_part_nonempty: If no delimiter is found and this\n                                            is true the contents of the string\n                                            will be returned in the second part\n                                            of the tuple instead of the first\n                                            one.\n        :return:                            (first_part, second_part)\n        '
        temp_string = string.replace('\\\\', 'oo')
        i = temp_string.find('\\')
        while i != -1:
            temp_string = temp_string[:i] + 'oo' + temp_string[i + 2:]
            i = temp_string.find('\\', i + 2)
        delim_pos = len(string)
        used_delim = ''
        for delim in delimiters:
            pos = temp_string.find(delim)
            if 0 <= pos < delim_pos:
                delim_pos = pos
                used_delim = delim
        if return_second_part_nonempty and delim_pos == len(string):
            return ('', string.strip(' \n'))
        first_part = string[:delim_pos]
        second_part = string[delim_pos + (len(used_delim) if strip_delim else 0):]
        if not position_is_escaped(second_part, len(second_part) - 1):
            first_part = unescaped_rstrip(first_part)
            second_part = unescaped_rstrip(second_part)
        return (first_part.lstrip().rstrip('\n'), second_part.lstrip().rstrip('\n'))

    def __get_section_name(self, line):
        if False:
            while True:
                i = 10
        for (begin, end) in self.section_name_surroundings.items():
            if line[0:len(begin)] == begin and line[len(line) - len(end):len(line)] == end:
                return line[len(begin):len(line) - len(end)].strip(' \n')
        return ''

    def __extract_keys_and_value(self, line, delimiters, return_second_part_nonempty=False):
        if False:
            while True:
                i = 10
        '\n        This method extracts the keys and values from the give string by\n        splitting them based on the delimiters provided.\n\n        :param line:                        The input string.\n        :param delimiters:                  A list of delimiters to split the\n                                            strings on.\n        :param return_second_part_nonempty: If no delimiter is found and this\n                                            is true the contents of the string\n                                            will be returned as value\n        :return:                            The parsed keys and values from a\n                                            line.\n        '
        (key_part, value) = self.__separate_by_first_occurrence(line, delimiters, True, return_second_part_nonempty)
        keys = list(StringConverter(key_part, list_delimiters=self.key_delimiters).__iter__(remove_backslashes=False))
        return (keys, value)