from coalib.bearlib.abstractions.SectionCreatable import SectionCreatable
from coala_utils.decorators import enforce_signature

class SpacingHelper(SectionCreatable):
    DEFAULT_TAB_WIDTH = 4

    def __init__(self, tab_width: int=DEFAULT_TAB_WIDTH):
        if False:
            i = 10
            return i + 15
        '\n        Creates a helper object for spacing operations.\n\n        :param tab_width: The number of spaces which visually equals a tab.\n        '
        SectionCreatable.__init__(self)
        if not isinstance(tab_width, int):
            raise TypeError("The 'tab_width' parameter should be an integer.")
        self.tab_width = tab_width

    @enforce_signature
    def get_indentation(self, line: str):
        if False:
            while True:
                i = 10
        '\n        Checks the lines indentation.\n\n        :param line: A string to check for indentation.\n        :return:     The indentation count in spaces.\n        '
        count = 0
        for char in line:
            if char == ' ':
                count += 1
                continue
            if char == '\t':
                count += self.tab_width - count % self.tab_width
                continue
            break
        return count

    @enforce_signature
    def replace_tabs_with_spaces(self, line: str):
        if False:
            print('Hello World!')
        '\n        Replaces tabs in this line with the appropriate number of spaces.\n\n        Example: " \t" will be converted to "    ", assuming the tab_width is\n        set to 4.\n\n        :param line: The string with tabs to replace.\n        :return:     A string with no tabs.\n        '
        for (t_position, t_length) in sorted(self.yield_tab_lengths(line), reverse=True):
            line = line[:t_position] + t_length * ' ' + line[t_position + 1:]
        return line

    @enforce_signature
    def yield_tab_lengths(self, input: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Yields position and size of tabs in a input string.\n\n        :param input: The string with tabs.\n        '
        tabless_position = 0
        for (index, char) in enumerate(input):
            if char == '\t':
                space_count = self.tab_width - tabless_position % self.tab_width
                yield (index, space_count)
                tabless_position += space_count
                continue
            tabless_position += 1

    @enforce_signature
    def replace_spaces_with_tabs(self, line: str):
        if False:
            return 10
        '\n        Replaces spaces with tabs where possible. However in no case only one\n        space will be replaced by a tab.\n\n        Example: " \t   a_text   another" will be converted to\n        "\t   a_text\tanother", assuming the tab_width is set to 4.\n\n        :param line: The string with spaces to replace.\n        :return:     The converted string.\n        '
        currspaces = 0
        result = ''
        tabless_position = 0
        for char in line:
            if char == ' ':
                currspaces += 1
                tabless_position += 1
            elif char == '\t':
                space_count = self.tab_width - tabless_position % self.tab_width
                currspaces += space_count
                tabless_position += space_count
            else:
                result += currspaces * ' ' + char
                currspaces = 0
                tabless_position += 1
            if tabless_position % self.tab_width == 0 and currspaces:
                if currspaces == 1 and char == ' ':
                    result += ' '
                else:
                    result += '\t'
                currspaces = 0
        result += currspaces * ' '
        return result