class GetCharMap:
    """

    Return the character map for the given value

    """

    def __init__(self, bug_handler, char_file):
        if False:
            i = 10
            return i + 15
        "\n\n        Required:\n\n            'char_file'--the file with the mappings\n\n        Returns:\n\n            nothing\n\n            "
        self.__char_file = char_file
        self.__bug_handler = bug_handler

    def get_char_map(self, map):
        if False:
            while True:
                i = 10
        found_map = False
        map_dict = {}
        self.__char_file.seek(0)
        for line in self.__char_file:
            if not line.strip():
                continue
            begin_element = '<%s>' % map
            end_element = '</%s>' % map
            if not found_map:
                if begin_element in line:
                    found_map = True
            else:
                if end_element in line:
                    break
                fields = line.split(':')
                fields[1].replace('\\colon', ':')
                map_dict[fields[1]] = fields[3]
        if not found_map:
            msg = 'no map found\nmap is "%s"\n' % (map,)
            raise self.__bug_handler(msg)
        return map_dict