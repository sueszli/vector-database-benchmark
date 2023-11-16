class TextWrap:
    """ Text wrap """
    char_widths = [(126, 1), (159, 0), (687, 1), (710, 0), (711, 1), (727, 0), (733, 1), (879, 0), (1154, 1), (1161, 0), (4347, 1), (4447, 2), (7467, 1), (7521, 0), (8369, 1), (8426, 0), (9000, 1), (9002, 2), (11021, 1), (12350, 2), (12351, 1), (12438, 2), (12442, 0), (19893, 2), (19967, 1), (55203, 2), (63743, 1), (64106, 2), (65039, 1), (65059, 0), (65131, 2), (65279, 1), (65376, 2), (65500, 1), (65510, 2), (120831, 1), (262141, 2), (1114109, 1)]

    @classmethod
    def get_width(cls, char):
        if False:
            while True:
                i = 10
        'Return the screen column width for a char'
        o = ord(char)
        if o == 14 or o == 15:
            return 0
        for (num, wid) in cls.char_widths:
            if o <= num:
                return wid
        return 1

    @classmethod
    def wrap(cls, text: str, width: int, once=True):
        if False:
            print('Hello World!')
        ' Wrap according to string length\n\n        Parameters\n        ----------\n        text: str\n            the text to be wrapped\n\n        width: int\n            the maximum length of a single line, the length of Chinese characters is 2\n\n        once: bool\n            whether to wrap only once\n\n        Returns\n        -------\n        wrap_text: str\n            text after auto word wrap process\n\n        is_wrapped: bool\n            whether a line break occurs in the text\n        '
        texts = text.strip().split('\n')
        result = []
        is_wrapped = False
        for text in texts:
            (text_wrapped, wrapped) = cls._wrap_line(text, width, once)
            is_wrapped |= wrapped
            result.append(text_wrapped)
            if once:
                result.extend(texts[1:])
                break
        return ('\n'.join(result), is_wrapped)

    @classmethod
    def _wrap_line(cls, text: str, width: int, once=True):
        if False:
            while True:
                i = 10
        count = 0
        last_count = 0
        chars = []
        is_wrapped = False
        break_pos = 0
        is_break_alpha = True
        n_inside_break = 0
        i = 0
        while i < len(text):
            c = text[i]
            length = cls.get_width(c)
            count += length
            if c == ' ' or length > 1:
                break_pos = i + n_inside_break
                last_count = count
                is_break_alpha = length == 1
            if count <= width:
                chars.append(c)
                i += 1
                continue
            if break_pos > 0 and is_break_alpha:
                if c != ' ':
                    chars[break_pos] = '\n'
                    chars.append(c)
                    if last_count != 0:
                        count -= last_count
                        last_count = 0
                    else:
                        chars.insert(i, '\n')
                        break_pos = i
                        n_inside_break += 1
                else:
                    chars.append('\n')
                    count = 0
                    last_count = 0
            else:
                chars.extend(('\n', c))
                count = length
            is_wrapped = True
            if once:
                return (''.join(chars) + text[i + 1:], True)
            i += 1
        return (''.join(chars), is_wrapped)