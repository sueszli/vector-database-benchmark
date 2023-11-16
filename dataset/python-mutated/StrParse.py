import sys
import string
if sys.version_info >= (3, 0):

    def unichr(character):
        if False:
            print('Hello World!')
        return chr(character)

def ConvertNEXTSTEPToUnicode(hex_digits):
    if False:
        print('Hello World!')
    conversion = {'80': 'a0', '81': 'c0', '82': 'c1', '83': 'c2', '84': 'c3', '85': 'c4', '86': 'c5', '87': 'c7', '88': 'c8', '89': 'c9', '8a': 'ca', '8b': 'cb', '8c': 'cc', '8d': 'cd', '8e': 'ce', '8f': 'cf', '90': 'd0', '91': 'd1', '92': 'd2', '93': 'd3', '94': 'd4', '95': 'd5', '96': 'd6', '97': 'd9', '98': 'da', '99': 'db', '9a': 'dc', '9b': 'dd', '9c': 'de', '9d': 'b5', '9e': 'd7', '9f': 'f7', 'a0': 'a9', 'a1': 'a1', 'a2': 'a2', 'a3': 'a3', 'a4': '44', 'a5': 'a5', 'a6': '92', 'a7': 'a7', 'a8': 'a4', 'a9': '19', 'aa': '1c', 'ab': 'ab', 'ac': '39', 'ad': '3a', 'ae': '01', 'af': '02', 'b0': 'ae', 'b1': '13', 'b2': '20', 'b3': '21', 'b4': 'b7', 'b5': 'a6', 'b6': 'b6', 'b7': '22', 'b8': '1a', 'b9': '1e', 'ba': '1d', 'bb': 'bb', 'bc': '26', 'bd': '30', 'be': 'ac', 'bf': 'bf', 'c0': 'b9', 'c1': 'cb', 'c2': 'b4', 'c3': 'c6', 'c4': 'dc', 'c5': 'af', 'c6': 'd8', 'c7': 'd9', 'c8': 'a8', 'c9': 'b2', 'ca': 'da', 'cb': 'b8', 'cc': 'b3', 'cd': 'dd', 'ce': 'db', 'cf': 'c7', 'd0': '14', 'd1': 'b1', 'd2': 'bc', 'd3': 'bd', 'd4': 'be', 'd5': 'e0', 'd6': 'e1', 'd7': 'e2', 'd8': 'e3', 'd9': 'e4', 'da': 'e5', 'db': 'e7', 'dc': 'e8', 'dd': 'e9', 'de': 'ea', 'df': 'eb', 'e0': 'ec', 'e1': 'c6', 'e2': 'ed', 'e3': 'aa', 'e4': 'ee', 'e5': 'ef', 'e6': 'f0', 'e7': 'f1', 'e8': '41', 'e9': 'd8', 'ea': '52', 'eb': 'ba', 'ec': 'f2', 'ed': 'f3', 'ee': 'f4', 'ef': 'f5', 'f0': 'f6', 'f1': 'e6', 'f2': 'f9', 'f3': 'fa', 'f4': 'fb', 'f5': '31', 'f6': 'fc', 'f7': 'fd', 'f8': '42', 'f9': 'f8', 'fa': '53', 'fb': 'df', 'fc': 'fe', 'fd': 'ff', 'fe': 'fd', 'ff': 'fd'}
    return conversion[hex_digits]

def IsOctalNumber(character):
    if False:
        while True:
            i = 10
    oct_digits = set(string.octdigits)
    return set(character).issubset(oct_digits)

def IsHexNumber(character):
    if False:
        i = 10
        return i + 15
    hex_digits = set(string.hexdigits)
    return set(character).issubset(hex_digits)

def SanitizeCharacter(character):
    if False:
        return 10
    char = character
    escaped_characters = {'\x07': '\\a', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x0b': '\\v', '"': '\\"'}
    if character in escaped_characters.keys():
        char = escaped_characters[character]
    return char

def UnQuotifyString(string_data, start_index, end_index):
    if False:
        for i in range(10):
            print('nop')
    formatted_string = ''
    extracted_string = string_data[start_index:end_index]
    string_length = len(extracted_string)
    all_cases = ['0', '1', '2', '3', '4', '5', '6', '7', 'a', 'b', 'f', 'n', 'r', 't', 'v', '"', '\n', 'U']
    index = 0
    while index < string_length:
        current_char = extracted_string[index]
        if current_char == '\\':
            next_char = extracted_string[index + 1]
            if next_char in all_cases:
                index += 1
                if next_char == 'a':
                    formatted_string += '\x07'
                if next_char == 'b':
                    formatted_string += '\x08'
                if next_char == 'f':
                    formatted_string += '\x0c'
                if next_char == 'n':
                    formatted_string += '\n'
                if next_char == 'r':
                    formatted_string += '\r'
                if next_char == 't':
                    formatted_string += '\t'
                if next_char == 'v':
                    formatted_string += '\x0b'
                if next_char == '"':
                    formatted_string += '"'
                if next_char == '\n':
                    formatted_string += '\n'
                if next_char == 'U':
                    starting_index = index + 1
                    ending_index = starting_index + 4
                    unicode_numbers = extracted_string[starting_index:ending_index]
                    for number in unicode_numbers:
                        index += 1
                        if IsHexNumber(number) is False:
                            message = 'Invalid unicode sequence on line ' + str(LineNumberForIndex(string_data, start_index + index))
                            raise Exception(message)
                    formatted_string += unichr(int(unicode_numbers, 16))
                if IsOctalNumber(next_char) is True:
                    starting_index = index
                    ending_index = starting_index + 1
                    for oct_index in range(3):
                        test_index = starting_index + oct_index
                        test_oct = extracted_string[test_index]
                        if IsOctalNumber(test_oct) is True:
                            ending_index += 1
                    octal_numbers = extracted_string[starting_index:ending_index]
                    hex_number = int(octal_numbers, 8)
                    hex_str = format(hex_number, 'x')
                    if hex_number >= 128:
                        hex_str = ConvertNEXTSTEPToUnicode(hex_str)
                    formatted_string += unichr(int('00' + hex_str, 16))
            else:
                formatted_string += current_char
                index += 1
                formatted_string += next_char
        else:
            formatted_string += current_char
        index += 1
    return formatted_string

def LineNumberForIndex(string_data, current_index):
    if False:
        i = 10
        return i + 15
    line_number = 1
    index = 0
    string_length = len(string_data)
    while index < current_index and index < string_length:
        current_char = string_data[index]
        if IsNewline(current_char) is True:
            line_number += 1
        index += 1
    return line_number

def IsValidUnquotedStringCharacter(character):
    if False:
        while True:
            i = 10
    if len(character) == 1:
        valid_characters = set(string.ascii_letters + string.digits + '_$/:.-')
        return set(character).issubset(valid_characters)
    else:
        message = 'The function "IsValidUnquotedStringCharacter()" can only take single characters!'
        raise ValueError(message)

def IsSpecialWhitespace(character):
    if False:
        for i in range(10):
            print('nop')
    value = ord(character)
    result = value >= 9 and value <= 13
    return result

def IsUnicodeSeparator(character):
    if False:
        while True:
            i = 10
    value = ord(character)
    result = value == 8232 or value == 8233
    return result

def IsRegularWhitespace(character):
    if False:
        return 10
    value = ord(character)
    result = value == 32 or IsUnicodeSeparator(character)
    return result

def IsDataFormattingWhitespace(character):
    if False:
        i = 10
        return i + 15
    value = ord(character)
    result = IsNewline(character) or IsRegularWhitespace(character) or value == 9
    return result

def IsNewline(character):
    if False:
        return 10
    value = ord(character)
    result = value == 13 or value == 10
    return result

def IsEndOfLine(character):
    if False:
        i = 10
        return i + 15
    result = IsNewline(character) or IsUnicodeSeparator(character)
    return result

def IndexOfNextNonSpace(string_data, current_index):
    if False:
        for i in range(10):
            print('nop')
    successful = False
    found_index = current_index
    string_length = len(string_data)
    annotation_string = ''
    while found_index < string_length:
        current_char = string_data[found_index]
        if IsSpecialWhitespace(current_char) is True:
            found_index += 1
            continue
        if IsRegularWhitespace(current_char) is True:
            found_index += 1
            continue
        if current_char == '/':
            next_index = found_index + 1
            if next_index >= string_length:
                successful = True
                break
            else:
                next_character = string_data[next_index]
                if next_character == '/':
                    found_index += 1
                    next_index = found_index
                    first_pass = True
                    while next_index < string_length:
                        test_char = string_data[next_index]
                        if IsEndOfLine(test_char) is True:
                            break
                        elif first_pass is False:
                            annotation_string += test_char
                        else:
                            first_pass = False
                        next_index += 1
                    found_index = next_index
                elif next_character == '*':
                    found_index += 1
                    next_index = found_index
                    first_pass = True
                    while next_index < string_length:
                        test_char = string_data[next_index]
                        if test_char == '*' and next_index + 1 < string_length and (string_data[next_index + 1] == '/'):
                            next_index += 2
                            break
                        elif first_pass != True:
                            annotation_string += test_char
                        else:
                            first_pass = False
                        next_index += 1
                    found_index = next_index
                else:
                    successful = True
                    break
        else:
            successful = True
            break
    result = (successful, found_index, annotation_string)
    return result