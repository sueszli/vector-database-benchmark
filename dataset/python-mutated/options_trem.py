import sys

class ParseOptions:
    """
        Requires:
           system_string --The string from the command line
           options_dict -- a dictionary with the key equal to the opition, and
           a list describing that option. (See below)
        Returns:
            A tuple. The first item in the tuple is a dictionary containing
            the arguments for each options. The second is a list of the
            arguments.
            If invalid options are passed to the module, 0,0 is returned.
        Examples:
            Your script has the option '--indents', and '--output=file'.
            You want to give short option names as well:
                --i and -o=file
            Use this:
                options_dict = {'output':   [1, 'o'],
                                'indents':  [0, 'i']
                                }
                options_obj = ParseOptions(
                                                system_string = sys.argv,
                                                options_dict = options_dict
                        )
                options, arguments = options_obj.parse_options()
                print options
                print arguments
            The result will be:
                {indents:None, output:'/home/paul/file'}, ['/home/paul/input']
        """

    def __init__(self, system_string, options_dict):
        if False:
            for i in range(10):
                print('nop')
        self.__system_string = system_string[1:]
        long_list = self.__make_long_list_func(options_dict)
        short_list = self.__make_short_list_func(options_dict)
        self.__legal_options = long_list + short_list
        self.__short_long_dict = self.__make_short_long_dict_func(options_dict)
        self.__opt_with_args = self.__make_options_with_arg_list(options_dict)
        self.__options_okay = 1

    def __make_long_list_func(self, options_dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Required:\n            options_dict -- the dictionary mapping options to a list\n        Returns:\n            a list of legal options\n        '
        legal_list = []
        keys = options_dict.keys()
        for key in keys:
            key = '--' + key
            legal_list.append(key)
        return legal_list

    def __make_short_list_func(self, options_dict):
        if False:
            print('Hello World!')
        '\n        Required:\n            options_dict --the dictionary mapping options to a list\n        Returns:\n            a list of legal short options\n        '
        legal_list = []
        keys = options_dict.keys()
        for key in keys:
            values = options_dict[key]
            try:
                legal_list.append('-' + values[1])
            except IndexError:
                pass
        return legal_list

    def __make_short_long_dict_func(self, options_dict):
        if False:
            i = 10
            return i + 15
        '\n        Required:\n            options_dict --the dictionary mapping options to a list\n        Returns:\n            a dictionary with keys of short options and values of long options\n        Logic:\n            read through the options dictionary and pair short options with long options\n        '
        short_long_dict = {}
        keys = options_dict.keys()
        for key in keys:
            values = options_dict[key]
            try:
                short = '-' + values[1]
                long = '--' + key
                short_long_dict[short] = long
            except IndexError:
                pass
        return short_long_dict

    def __make_options_with_arg_list(self, options_dict):
        if False:
            print('Hello World!')
        '\n        Required:\n            options_dict --the dictionary mapping options to a list\n        Returns:\n            a list of options that take arguments.\n        '
        opt_with_arg = []
        keys = options_dict.keys()
        for key in keys:
            values = options_dict[key]
            try:
                if values[0]:
                    opt_with_arg.append('--' + key)
            except IndexError:
                pass
        return opt_with_arg

    def __sub_short_with_long(self):
        if False:
            return 10
        '\n        Required:\n            nothing\n        Returns:\n            a new system string\n        Logic:\n            iterate through the system string and replace short options with long options\n        '
        new_string = []
        sub_list = self.__short_long_dict.keys()
        for item in self.__system_string:
            if item in sub_list:
                item = self.__short_long_dict[item]
            new_string.append(item)
        return new_string

    def __pair_arg_with_option(self):
        if False:
            print('Hello World!')
        "\n        Required:\n            nothing\n        Returns\n            nothing (changes value of self.__system_string)\n        Logic:\n            iterate through the system string, and match arguments with options:\n                old_list = ['--foo', 'bar']\n                new_list = ['--foo=bar'\n        "
        opt_len = len(self.__system_string)
        new_system_string = []
        counter = 0
        slurp_value = 0
        for arg in self.__system_string:
            counter += 1
            if slurp_value:
                slurp_value = 0
                continue
            if arg[0] != '-':
                new_system_string.append(arg)
            elif '=' in arg:
                new_system_string.append(arg)
            elif arg in self.__opt_with_args:
                if counter + 1 > opt_len:
                    sys.stderr.write('option "%s" must take an argument\n' % arg)
                    new_system_string.append(arg)
                    self.__options_okay = 0
                elif self.__system_string[counter][0] == '-':
                    sys.stderr.write('option "%s" must take an argument\n' % arg)
                    new_system_string.append(arg)
                    self.__options_okay = 0
                else:
                    new_system_string.append(arg + '=' + self.__system_string[counter])
                    slurp_value = 1
            else:
                new_system_string.append(arg)
        return new_system_string

    def __get_just_options(self):
        if False:
            while True:
                i = 10
        '\n        Requires:\n            nothing\n        Returns:\n            list of options\n        Logic:\n            Iterate through the self.__system string, looking for the last\n            option. The options are everything in the system string before the\n            last option.\n            Check to see that the options contain no arguments.\n        '
        highest = 0
        counter = 0
        found_options = 0
        for item in self.__system_string:
            if item[0] == '-':
                highest = counter
                found_options = 1
            counter += 1
        if found_options:
            just_options = self.__system_string[:highest + 1]
            arguments = self.__system_string[highest + 1:]
        else:
            just_options = []
            arguments = self.__system_string
        if found_options:
            for item in just_options:
                if item[0] != '-':
                    sys.stderr.write('%s is an argument in an option list\n' % item)
                    self.__options_okay = 0
        return (just_options, arguments)

    def __is_legal_option_func(self):
        if False:
            while True:
                i = 10
        '\n        Requires:\n            nothing\n        Returns:\n            nothing\n        Logic:\n            Check each value in the newly creatd options list to see if it\n            matches what the user describes as a legal option.\n        '
        illegal_options = []
        for arg in self.__system_string:
            if '=' in arg:
                temp_list = arg.split('=')
                arg = temp_list[0]
            if arg not in self.__legal_options and arg[0] == '-':
                illegal_options.append(arg)
        if illegal_options:
            self.__options_okay = 0
            sys.stderr.write('The following options are not permitted:\n')
            for not_legal in illegal_options:
                sys.stderr.write('%s\n' % not_legal)

    def __make_options_dict(self, options):
        if False:
            while True:
                i = 10
        options_dict = {}
        for item in options:
            if '=' in item:
                (option, arg) = item.split('=')
            else:
                option = item
                arg = None
            if option[0] == '-':
                option = option[1:]
            if option[0] == '-':
                option = option[1:]
            options_dict[option] = arg
        return options_dict

    def parse_options(self):
        if False:
            i = 10
            return i + 15
        self.__system_string = self.__sub_short_with_long()
        self.__system_string = self.__pair_arg_with_option()
        (options, arguments) = self.__get_just_options()
        self.__is_legal_option_func()
        if self.__options_okay:
            options_dict = self.__make_options_dict(options)
            return (options_dict, arguments)
        else:
            return (0, 0)
if __name__ == '__main__':
    this_dict = {'indents': [0, 'i'], 'output': [1, 'o'], 'test3': [1, 't']}
    test_obj = ParseOptions(system_string=sys.argv, options_dict=this_dict)
    (options, the_args) = test_obj.parse_options()
    print(options, the_args)
    "\n    this_options = ['--foo', '-o']\n    this_opt_with_args = ['--foo']\n    "