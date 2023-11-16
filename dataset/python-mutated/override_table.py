class OverrideTable:
    """
    Parse a line of text to make the override table. Return a string
    (which will convert to XML) and the dictionary containing all the
    information about the lists. This dictionary is the result of the
    dictionary that is first passed to this module. This module
    modifies the dictionary, assigning lists numbers to each list.
    """

    def __init__(self, list_of_lists, run_level=1):
        if False:
            while True:
                i = 10
        self.__list_of_lists = list_of_lists
        self.__initiate_values()
        self.__run_level = run_level

    def __initiate_values(self):
        if False:
            print('Hello World!')
        self.__override_table_final = ''
        self.__state = 'default'
        self.__override_list = []
        self.__state_dict = {'default': self.__default_func, 'override': self.__override_func, 'unsure_ob': self.__after_bracket_func}
        self.__override_dict = {'cw<ls<lis-tbl-id': 'list-table-id', 'cw<ls<list-id___': 'list-id'}

    def __override_func(self, line):
        if False:
            return 10
        '\n        Requires:\n            line -- line to parse\n        Returns:\n            nothing\n        Logic:\n            The group {\\override has been found.\n            Check for the end of the group.\n            Otherwise, add appropriate tokens to the override dictionary.\n        '
        if self.__token_info == 'cb<nu<clos-brack' and self.__cb_count == self.__override_ob_count:
            self.__state = 'default'
            self.__parse_override_dict()
        else:
            att = self.__override_dict.get(self.__token_info)
            if att:
                value = line[20:]
                self.__override_list[-1][att] = value

    def __parse_override_dict(self):
        if False:
            print('Hello World!')
        '\n        Requires:\n            nothing\n        Returns:\n            nothing\n        Logic:\n            The list of all information about RTF lists has been passed to\n            this module. As of this point, this python list has no id number,\n            which is needed later to identify which lists in the body should\n            be assigned which formatting commands from the list-table.\n            In order to get an id, I have to check to see when the list-table-id\n            from the override_dict (generated in this module) matches the list-table-id\n            in list_of_lists (generated in the list_table.py module). When a match is found,\n            append the lists numbers to the self.__list_of_lists dictionary\n            that contains the empty lists:\n                [[{list-id:[HERE!],[{}]]\n            This is a list, since one list in the table in the preamble of RTF can\n            apply to multiple lists in the body.\n        '
        override_dict = self.__override_list[-1]
        list_id = override_dict.get('list-id')
        if list_id is None and self.__level > 3:
            msg = 'This override does not appear to have a list-id\n'
            raise self.__bug_handler(msg)
        current_table_id = override_dict.get('list-table-id')
        if current_table_id is None and self.__run_level > 3:
            msg = 'This override does not appear to have a list-table-id\n'
            raise self.__bug_handler(msg)
        counter = 0
        for list in self.__list_of_lists:
            info_dict = list[0]
            old_table_id = info_dict.get('list-table-id')
            if old_table_id == current_table_id:
                self.__list_of_lists[counter][0]['list-id'].append(list_id)
                break
            counter += 1

    def __parse_lines(self, line):
        if False:
            while True:
                i = 10
        '\n        Requires:\n            line --ine to parse\n        Returns:\n            nothing\n        Logic:\n            Break the into tokens by splitting it on the newline.\n            Call on the method according to the state.\n        '
        lines = line.split('\n')
        self.__ob_count = 0
        self.__ob_group = 0
        for line in lines:
            self.__token_info = line[:16]
            if self.__token_info == 'ob<nu<open-brack':
                self.__ob_count = line[-4:]
                self.__ob_group += 1
            if self.__token_info == 'cb<nu<clos-brack':
                self.__cb_count = line[-4:]
                self.__ob_group -= 1
            action = self.__state_dict.get(self.__state)
            if action is None:
                print(self.__state)
            action(line)
        self.__write_final_string()

    def __default_func(self, line):
        if False:
            return 10
        '\n        Requires:\n            line -- line to parse\n        Return:\n            nothing\n        Logic:\n            Look for an open bracket and change states when found.\n        '
        if self.__token_info == 'ob<nu<open-brack':
            self.__state = 'unsure_ob'

    def __after_bracket_func(self, line):
        if False:
            while True:
                i = 10
        '\n        Requires:\n            line -- line to parse\n        Returns:\n            nothing\n        Logic:\n            The last token was an open bracket. You need to determine\n            the group based on the token after.\n            WARNING: this could cause problems. If no group is found, the\n            state will remain unsure_ob, which means no other text will be\n            parsed. I should do states by a list and simply pop this\n            unsure_ob state to get the previous state.\n        '
        if self.__token_info == 'cw<ls<lis-overid':
            self.__state = 'override'
            self.__override_ob_count = self.__ob_count
            the_dict = {}
            self.__override_list.append(the_dict)
        elif self.__run_level > 3:
            msg = 'No matching token after open bracket\n'
            msg += 'token is "%s\n"' % line
            raise self.__bug_handler(msg)

    def __write_final_string(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Requires:\n            line -- line to parse\n        Returns:\n            nothing\n        Logic:\n            First write out the override-table tag.\n            Iteratere through the dictionaries in the main override_list.\n            For each dictionary, write an empty tag "override-list". Add\n            the attributes and values of the tag from the dictionary.\n        '
        self.__override_table_final = 'mi<mk<over_beg_\n'
        self.__override_table_final += 'mi<tg<open______<override-table\n' + 'mi<mk<overbeg__\n' + self.__override_table_final
        for the_dict in self.__override_list:
            self.__override_table_final += 'mi<tg<empty-att_<override-list'
            the_keys = the_dict.keys()
            for the_key in the_keys:
                self.__override_table_final += f'<{the_key}>{the_dict[the_key]}'
            self.__override_table_final += '\n'
        self.__override_table_final += '\n'
        self.__override_table_final += 'mi<mk<overri-end\n' + 'mi<tg<close_____<override-table\n'
        self.__override_table_final += 'mi<mk<overribend_\n'

    def parse_override_table(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Requires:\n            line -- line with border definition in it\n        Returns:\n            A string that will be converted to XML, and a dictionary of\n            all the properties of the RTF lists.\n        Logic:\n        '
        self.__parse_lines(line)
        return (self.__override_table_final, self.__list_of_lists)