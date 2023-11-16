import copy
from ast import literal_eval
from robot.api import logger
from robot.utils import get_error_message, is_dict_like, is_list_like, is_truthy, Matcher, NOT_SET, plural_or_not as s, seq2str, seq2str2, type_name
from robot.utils.asserts import assert_equal
from robot.version import get_version

class _List:

    def convert_to_list(self, item):
        if False:
            return 10
        'Converts the given ``item`` to a Python ``list`` type.\n\n        Mainly useful for converting tuples and other iterable to lists.\n        Use `Create List` from the BuiltIn library for constructing new lists.\n        '
        return list(item)

    def append_to_list(self, list_, *values):
        if False:
            return 10
        "Adds ``values`` to the end of ``list``.\n\n        Example:\n        | Append To List | ${L1} | xxx |   |   |\n        | Append To List | ${L2} | x   | y | z |\n        =>\n        | ${L1} = ['a', 'xxx']\n        | ${L2} = ['a', 'b', 'x', 'y', 'z']\n        "
        self._validate_list(list_)
        for value in values:
            list_.append(value)

    def insert_into_list(self, list_, index, value):
        if False:
            while True:
                i = 10
        "Inserts ``value`` into ``list`` to the position specified with ``index``.\n\n        Index ``0`` adds the value into the first position, ``1`` to the second,\n        and so on. Inserting from right works with negative indices so that\n        ``-1`` is the second last position, ``-2`` third last, and so on. Use\n        `Append To List` to add items to the end of the list.\n\n        If the absolute value of the index is greater than\n        the length of the list, the value is added at the end\n        (positive index) or the beginning (negative index). An index\n        can be given either as an integer or a string that can be\n        converted to an integer.\n\n        Example:\n        | Insert Into List | ${L1} | 0     | xxx |\n        | Insert Into List | ${L2} | ${-1} | xxx |\n        =>\n        | ${L1} = ['xxx', 'a']\n        | ${L2} = ['a', 'xxx', 'b']\n        "
        self._validate_list(list_)
        list_.insert(self._index_to_int(index), value)

    def combine_lists(self, *lists):
        if False:
            i = 10
            return i + 15
        "Combines the given ``lists`` together and returns the result.\n\n        The given lists are not altered by this keyword.\n\n        Example:\n        | ${x} = | Combine Lists | ${L1} | ${L2} |       |\n        | ${y} = | Combine Lists | ${L1} | ${L2} | ${L1} |\n        =>\n        | ${x} = ['a', 'a', 'b']\n        | ${y} = ['a', 'a', 'b', 'a']\n        | ${L1} and ${L2} are not changed.\n        "
        self._validate_lists(*lists)
        ret = []
        for item in lists:
            ret.extend(item)
        return ret

    def set_list_value(self, list_, index, value):
        if False:
            print('Hello World!')
        "Sets the value of ``list`` specified by ``index`` to the given ``value``.\n\n        Index ``0`` means the first position, ``1`` the second and so on.\n        Similarly, ``-1`` is the last position, ``-2`` second last, and so on.\n        Using an index that does not exist on the list causes an error.\n        The index can be either an integer or a string that can be converted to\n        an integer.\n\n        Example:\n        | Set List Value | ${L3} | 1  | xxx |\n        | Set List Value | ${L3} | -1 | yyy |\n        =>\n        | ${L3} = ['a', 'xxx', 'yyy']\n\n        Starting from Robot Framework 6.1, it is also possible to use the native\n        item assignment syntax. This is equivalent to the above:\n        | ${L3}[1] =  | Set Variable | xxx |\n        | ${L3}[-1] = | Set Variable | yyy |\n        "
        self._validate_list(list_)
        try:
            list_[self._index_to_int(index)] = value
        except IndexError:
            self._index_error(list_, index)

    def remove_values_from_list(self, list_, *values):
        if False:
            print('Hello World!')
        "Removes all occurrences of given ``values`` from ``list``.\n\n        It is not an error if a value does not exist in the list at all.\n\n        Example:\n        | Remove Values From List | ${L4} | a | c | e | f |\n        =>\n        | ${L4} = ['b', 'd']\n        "
        self._validate_list(list_)
        for value in values:
            while value in list_:
                list_.remove(value)

    def remove_from_list(self, list_, index):
        if False:
            print('Hello World!')
        "Removes and returns the value specified with an ``index`` from ``list``.\n\n        Index ``0`` means the first position, ``1`` the second and so on.\n        Similarly, ``-1`` is the last position, ``-2`` the second last, and so on.\n        Using an index that does not exist on the list causes an error.\n        The index can be either an integer or a string that can be converted\n        to an integer.\n\n        Example:\n        | ${x} = | Remove From List | ${L2} | 0 |\n        =>\n        | ${x} = 'a'\n        | ${L2} = ['b']\n        "
        self._validate_list(list_)
        try:
            return list_.pop(self._index_to_int(index))
        except IndexError:
            self._index_error(list_, index)

    def remove_duplicates(self, list_):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list without duplicates based on the given ``list``.\n\n        Creates and returns a new list that contains all items in the given\n        list so that one item can appear only once. Order of the items in\n        the new list is the same as in the original except for missing\n        duplicates. Number of the removed duplicates is logged.\n        '
        self._validate_list(list_)
        ret = []
        for item in list_:
            if item not in ret:
                ret.append(item)
        removed = len(list_) - len(ret)
        logger.info(f'{removed} duplicate{s(removed)} removed.')
        return ret

    def get_from_list(self, list_, index):
        if False:
            return 10
        "Returns the value specified with an ``index`` from ``list``.\n\n        The given list is never altered by this keyword.\n\n        Index ``0`` means the first position, ``1`` the second, and so on.\n        Similarly, ``-1`` is the last position, ``-2`` the second last, and so on.\n        Using an index that does not exist on the list causes an error.\n        The index can be either an integer or a string that can be converted\n        to an integer.\n\n        Examples (including Python equivalents in comments):\n        | ${x} = | Get From List | ${L5} | 0  | # L5[0]  |\n        | ${y} = | Get From List | ${L5} | -2 | # L5[-2] |\n        =>\n        | ${x} = 'a'\n        | ${y} = 'd'\n        | ${L5} is not changed\n        "
        self._validate_list(list_)
        try:
            return list_[self._index_to_int(index)]
        except IndexError:
            self._index_error(list_, index)

    def get_slice_from_list(self, list_, start=0, end=None):
        if False:
            return 10
        "Returns a slice of the given list between ``start`` and ``end`` indexes.\n\n        The given list is never altered by this keyword.\n\n        If both ``start`` and ``end`` are given, a sublist containing values\n        from ``start`` to ``end`` is returned. This is the same as\n        ``list[start:end]`` in Python. To get all items from the beginning,\n        use 0 as the start value, and to get all items until and including\n        the end, use ``None`` (default) as the end value.\n\n        Using ``start`` or ``end`` not found on the list is the same as using\n        the largest (or smallest) available index.\n\n        Examples (incl. Python equivalents in comments):\n        | ${x} = | Get Slice From List | ${L5} | 2      | 4 | # L5[2:4]    |\n        | ${y} = | Get Slice From List | ${L5} | 1      |   | # L5[1:None] |\n        | ${z} = | Get Slice From List | ${L5} | end=-2 |   | # L5[0:-2]   |\n        =>\n        | ${x} = ['c', 'd']\n        | ${y} = ['b', 'c', 'd', 'e']\n        | ${z} = ['a', 'b', 'c']\n        | ${L5} is not changed\n        "
        self._validate_list(list_)
        start = self._index_to_int(start, True)
        if end is not None:
            end = self._index_to_int(end)
        return list_[start:end]

    def count_values_in_list(self, list_, value, start=0, end=None):
        if False:
            while True:
                i = 10
        'Returns the number of occurrences of the given ``value`` in ``list``.\n\n        The search can be narrowed to the selected sublist by the ``start`` and\n        ``end`` indexes having the same semantics as with `Get Slice From List`\n        keyword. The given list is never altered by this keyword.\n\n        Example:\n        | ${x} = | Count Values In List | ${L3} | b |\n        =>\n        | ${x} = 1\n        | ${L3} is not changed\n        '
        self._validate_list(list_)
        return self.get_slice_from_list(list_, start, end).count(value)

    def get_index_from_list(self, list_, value, start=0, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the index of the first occurrence of the ``value`` on the list.\n\n        The search can be narrowed to the selected sublist by the ``start`` and\n        ``end`` indexes having the same semantics as with `Get Slice From List`\n        keyword. In case the value is not found, -1 is returned. The given list\n        is never altered by this keyword.\n\n        Example:\n        | ${x} = | Get Index From List | ${L5} | d |\n        =>\n        | ${x} = 3\n        | ${L5} is not changed\n        '
        self._validate_list(list_)
        if start == '':
            start = 0
        list_ = self.get_slice_from_list(list_, start, end)
        try:
            return int(start) + list_.index(value)
        except ValueError:
            return -1

    def copy_list(self, list_, deepcopy=False):
        if False:
            i = 10
            return i + 15
        'Returns a copy of the given list.\n\n        If the optional ``deepcopy`` is given a true value, the returned\n        list is a deep copy. New option in Robot Framework 3.1.2.\n\n        The given list is never altered by this keyword.\n        '
        self._validate_list(list_)
        if deepcopy:
            return copy.deepcopy(list_)
        return list_[:]

    def reverse_list(self, list_):
        if False:
            for i in range(10):
                print('nop')
        "Reverses the given list in place.\n\n        Note that the given list is changed and nothing is returned. Use\n        `Copy List` first, if you need to keep also the original order.\n\n        | Reverse List | ${L3} |\n        =>\n        | ${L3} = ['c', 'b', 'a']\n        "
        self._validate_list(list_)
        list_.reverse()

    def sort_list(self, list_):
        if False:
            print('Hello World!')
        'Sorts the given list in place.\n\n        Sorting fails if items in the list are not comparable with each others.\n        On Python 2 most objects are comparable, but on Python 3 comparing,\n        for example, strings with numbers is not possible.\n\n        Note that the given list is changed and nothing is returned. Use\n        `Copy List` first, if you need to keep also the original order.\n        '
        self._validate_list(list_)
        list_.sort()

    def list_should_contain_value(self, list_, value, msg=None):
        if False:
            while True:
                i = 10
        'Fails if the ``value`` is not found from ``list``.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_list(list_)
        _verify_condition(value in list_, f"{seq2str2(list_)} does not contain value '{value}'.", msg)

    def list_should_not_contain_value(self, list_, value, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the ``value`` is found from ``list``.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_list(list_)
        _verify_condition(value not in list_, f"{seq2str2(list_)} contains value '{value}'.", msg)

    def list_should_not_contain_duplicates(self, list_, msg=None):
        if False:
            return 10
        'Fails if any element in the ``list`` is found from it more than once.\n\n        The default error message lists all the elements that were found\n        from the ``list`` multiple times, but it can be overridden by giving\n        a custom ``msg``. All multiple times found items and their counts are\n        also logged.\n\n        This keyword works with all iterables that can be converted to a list.\n        The original iterable is never altered.\n        '
        self._validate_list(list_)
        if not isinstance(list_, list):
            list_ = list(list_)
        dupes = []
        for item in list_:
            if item not in dupes:
                count = list_.count(item)
                if count > 1:
                    logger.info(f"'{item}' found {count} times.")
                    dupes.append(item)
        if dupes:
            raise AssertionError(msg or f'{seq2str(dupes)} found multiple times.')

    def lists_should_be_equal(self, list1, list2, msg=None, values=True, names=None, ignore_order=False):
        if False:
            print('Hello World!')
        'Fails if given lists are unequal.\n\n        The keyword first verifies that the lists have equal lengths, and then\n        it checks are all their values equal. Possible differences between the\n        values are listed in the default error message like ``Index 4: ABC !=\n        Abc``. The types of the lists do not need to be the same. For example,\n        Python tuple and list with same content are considered equal.\n\n        The error message can be configured using ``msg`` and ``values``\n        arguments:\n        - If ``msg`` is not given, the default error message is used.\n        - If ``msg`` is given and ``values`` gets a value considered true\n          (see `Boolean arguments`), the error message starts with the given\n          ``msg`` followed by a newline and the default message.\n        - If ``msg`` is given and ``values``  is not given a true value,\n          the error message is just the given ``msg``.\n\n        The optional ``names`` argument can be used for naming the indices\n        shown in the default error message. It can either be a list of names\n        matching the indices in the lists or a dictionary where keys are\n        indices that need to be named. It is not necessary to name all indices.\n        When using a dictionary, keys can be either integers\n        or strings that can be converted to integers.\n\n        Examples:\n        | ${names} = | Create List | First Name | Family Name | Email |\n        | Lists Should Be Equal | ${people1} | ${people2} | names=${names} |\n        | ${names} = | Create Dictionary | 0=First Name | 2=Email |\n        | Lists Should Be Equal | ${people1} | ${people2} | names=${names} |\n\n        If the items in index 2 would differ in the above examples, the error\n        message would contain a row like ``Index 2 (email): name@foo.com !=\n        name@bar.com``.\n\n        The optional ``ignore_order`` argument can be used to ignore the order\n        of the elements in the lists. Using it requires items to be sortable.\n        This is new in Robot Framework 3.2.\n\n        Example:\n        | ${list1} = | Create List | apple | cherry | banana |\n        | ${list2} = | Create List | cherry | banana | apple |\n        | Lists Should Be Equal | ${list1} | ${list2} | ignore_order=True |\n        '
        self._validate_lists(list1, list2)
        len1 = len(list1)
        len2 = len(list2)
        _verify_condition(len1 == len2, f'Lengths are different: {len1} != {len2}', msg, values)
        names = self._get_list_index_name_mapping(names, len1)
        if ignore_order:
            list1 = sorted(list1)
            list2 = sorted(list2)
        diffs = '\n'.join(self._yield_list_diffs(list1, list2, names))
        _verify_condition(not diffs, f'Lists are different:\n{diffs}', msg, values)

    def _get_list_index_name_mapping(self, names, list_length):
        if False:
            while True:
                i = 10
        if not names:
            return {}
        if is_dict_like(names):
            return dict(((int(index), names[index]) for index in names))
        return dict(zip(range(list_length), names))

    def _yield_list_diffs(self, list1, list2, names):
        if False:
            for i in range(10):
                print('nop')
        for (index, (item1, item2)) in enumerate(zip(list1, list2)):
            name = f' ({names[index]})' if index in names else ''
            try:
                assert_equal(item1, item2, msg=f'Index {index}{name}')
            except AssertionError as err:
                yield str(err)

    def list_should_contain_sub_list(self, list1, list2, msg=None, values=True):
        if False:
            return 10
        'Fails if not all elements in ``list2`` are found in ``list1``.\n\n        The order of values and the number of values are not taken into\n        account.\n\n        See `Lists Should Be Equal` for more information about configuring\n        the error message with ``msg`` and ``values`` arguments.\n        '
        self._validate_lists(list1, list2)
        diffs = ', '.join((str(item) for item in list2 if item not in list1))
        _verify_condition(not diffs, f'Following values were not found from first list: {diffs}', msg, values)

    def log_list(self, list_, level='INFO'):
        if False:
            while True:
                i = 10
        'Logs the length and contents of the ``list`` using given ``level``.\n\n        Valid levels are TRACE, DEBUG, INFO (default), and WARN.\n\n        If you only want to the length, use keyword `Get Length` from\n        the BuiltIn library.\n        '
        self._validate_list(list_)
        logger.write('\n'.join(self._log_list(list_)), level)

    def _log_list(self, list_):
        if False:
            print('Hello World!')
        if not list_:
            yield 'List is empty.'
        elif len(list_) == 1:
            yield f'List has one item:\n{list_[0]}'
        else:
            yield f'List length is {len(list_)} and it contains following items:'
            for (index, item) in enumerate(list_):
                yield f'{index}: {item}'

    def _index_to_int(self, index, empty_to_zero=False):
        if False:
            while True:
                i = 10
        if empty_to_zero and (not index):
            return 0
        try:
            return int(index)
        except ValueError:
            raise ValueError(f"Cannot convert index '{index}' to an integer.")

    def _index_error(self, list_, index):
        if False:
            print('Hello World!')
        raise IndexError(f'Given index {index} is out of the range 0-{len(list_) - 1}.')

    def _validate_list(self, list_, position=1):
        if False:
            while True:
                i = 10
        if not is_list_like(list_):
            raise TypeError(f'Expected argument {position} to be a list or list-like, got {type_name(list_)} instead.')

    def _validate_lists(self, *lists):
        if False:
            i = 10
            return i + 15
        for (index, item) in enumerate(lists, start=1):
            self._validate_list(item, index)

class _Dictionary:

    def convert_to_dictionary(self, item):
        if False:
            print('Hello World!')
        "Converts the given ``item`` to a Python ``dict`` type.\n\n        Mainly useful for converting other mappings to normal dictionaries.\n        This includes converting Robot Framework's own ``DotDict`` instances\n        that it uses if variables are created using the ``&{var}`` syntax.\n\n        Use `Create Dictionary` from the BuiltIn library for constructing new\n        dictionaries.\n        "
        return dict(item)

    def set_to_dictionary(self, dictionary, *key_value_pairs, **items):
        if False:
            i = 10
            return i + 15
        "Adds the given ``key_value_pairs`` and/or ``items`` to the ``dictionary``.\n\n        If given items already exist in the dictionary, their values are updated.\n\n        It is easiest to specify items using the ``name=value`` syntax:\n        | Set To Dictionary | ${D1} | key=value | second=${2} |\n        =>\n        | ${D1} = {'a': 1, 'key': 'value', 'second': 2}\n\n        A limitation of the above syntax is that keys must be strings.\n        That can be avoided by passing keys and values as separate arguments:\n        | Set To Dictionary | ${D1} | key | value | ${2} | value 2 |\n        =>\n        | ${D1} = {'a': 1, 'key': 'value', 2: 'value 2'}\n\n        Starting from Robot Framework 6.1, it is also possible to use the native\n        item assignment syntax. This is equivalent to the above:\n        | ${D1}[key] =  | Set Variable | value |\n        | ${D1}[${2}] = | Set Variable | value 2 |\n        "
        self._validate_dictionary(dictionary)
        if len(key_value_pairs) % 2 != 0:
            raise ValueError('Adding data to a dictionary failed. There should be even number of key-value-pairs.')
        for i in range(0, len(key_value_pairs), 2):
            dictionary[key_value_pairs[i]] = key_value_pairs[i + 1]
        dictionary.update(items)
        return dictionary

    def remove_from_dictionary(self, dictionary, *keys):
        if False:
            i = 10
            return i + 15
        "Removes the given ``keys`` from the ``dictionary``.\n\n        If the given ``key`` cannot be found from the ``dictionary``, it\n        is ignored.\n\n        Example:\n        | Remove From Dictionary | ${D3} | b | x | y |\n        =>\n        | ${D3} = {'a': 1, 'c': 3}\n        "
        self._validate_dictionary(dictionary)
        for key in keys:
            if key in dictionary:
                value = dictionary.pop(key)
                logger.info(f"Removed item with key '{key}' and value '{value}'.")
            else:
                logger.info(f"Key '{key}' not found.")

    def pop_from_dictionary(self, dictionary, key, default=NOT_SET):
        if False:
            for i in range(10):
                print('nop')
        "Pops the given ``key`` from the ``dictionary`` and returns its value.\n\n        By default the keyword fails if the given ``key`` cannot be found from\n        the ``dictionary``. If optional ``default`` value is given, it will be\n        returned instead of failing.\n\n        Example:\n        | ${val}= | Pop From Dictionary | ${D3} | b |\n        =>\n        | ${val} = 2\n        | ${D3} = {'a': 1, 'c': 3}\n        "
        self._validate_dictionary(dictionary)
        if default is NOT_SET:
            self.dictionary_should_contain_key(dictionary, key)
            return dictionary.pop(key)
        return dictionary.pop(key, default)

    def keep_in_dictionary(self, dictionary, *keys):
        if False:
            return 10
        "Keeps the given ``keys`` in the ``dictionary`` and removes all other.\n\n        If the given ``key`` cannot be found from the ``dictionary``, it\n        is ignored.\n\n        Example:\n        | Keep In Dictionary | ${D5} | b | x | d |\n        =>\n        | ${D5} = {'b': 2, 'd': 4}\n        "
        self._validate_dictionary(dictionary)
        remove_keys = [k for k in dictionary if k not in keys]
        self.remove_from_dictionary(dictionary, *remove_keys)

    def copy_dictionary(self, dictionary, deepcopy=False):
        if False:
            while True:
                i = 10
        'Returns a copy of the given dictionary.\n\n        The ``deepcopy`` argument controls should the returned dictionary be\n        a [https://docs.python.org/library/copy.html|shallow or deep copy].\n        By default returns a shallow copy, but that can be changed by giving\n        ``deepcopy`` a true value (see `Boolean arguments`). This is a new\n        option in Robot Framework 3.1.2. Earlier versions always returned\n        shallow copies.\n\n        The given dictionary is never altered by this keyword.\n        '
        self._validate_dictionary(dictionary)
        if deepcopy:
            return copy.deepcopy(dictionary)
        return dictionary.copy()

    def get_dictionary_keys(self, dictionary, sort_keys=True):
        if False:
            i = 10
            return i + 15
        "Returns keys of the given ``dictionary`` as a list.\n\n        By default keys are returned in sorted order (assuming they are\n        sortable), but they can be returned in the original order by giving\n        ``sort_keys``  a false value (see `Boolean arguments`). Notice that\n        with Python 3.5 and earlier dictionary order is undefined unless using\n        ordered dictionaries.\n\n        The given ``dictionary`` is never altered by this keyword.\n\n        Example:\n        | ${sorted} =   | Get Dictionary Keys | ${D3} |\n        | ${unsorted} = | Get Dictionary Keys | ${D3} | sort_keys=False |\n        =>\n        | ${sorted} = ['a', 'b', 'c']\n        | ${unsorted} = ['b', 'a', 'c']   # Order depends on Python version.\n\n        ``sort_keys`` is a new option in Robot Framework 3.1.2. Earlier keys\n        were always sorted.\n        "
        self._validate_dictionary(dictionary)
        keys = dictionary.keys()
        if sort_keys:
            try:
                return sorted(keys)
            except TypeError:
                pass
        return list(keys)

    def get_dictionary_values(self, dictionary, sort_keys=True):
        if False:
            i = 10
            return i + 15
        'Returns values of the given ``dictionary`` as a list.\n\n        Uses `Get Dictionary Keys` to get keys and then returns corresponding\n        values. By default keys are sorted and values returned in that order,\n        but this can be changed by giving ``sort_keys`` a false value (see\n        `Boolean arguments`). Notice that with Python 3.5 and earlier\n        dictionary order is undefined unless using ordered dictionaries.\n\n        The given ``dictionary`` is never altered by this keyword.\n\n        Example:\n        | ${sorted} =   | Get Dictionary Values | ${D3} |\n        | ${unsorted} = | Get Dictionary Values | ${D3} | sort_keys=False |\n        =>\n        | ${sorted} = [1, 2, 3]\n        | ${unsorted} = [2, 1, 3]    # Order depends on Python version.\n\n        ``sort_keys`` is a new option in Robot Framework 3.1.2. Earlier values\n        were always sorted based on keys.\n        '
        self._validate_dictionary(dictionary)
        keys = self.get_dictionary_keys(dictionary, sort_keys=sort_keys)
        return [dictionary[k] for k in keys]

    def get_dictionary_items(self, dictionary, sort_keys=True):
        if False:
            return 10
        "Returns items of the given ``dictionary`` as a list.\n\n        Uses `Get Dictionary Keys` to get keys and then returns corresponding\n        items. By default keys are sorted and items returned in that order,\n        but this can be changed by giving ``sort_keys`` a false value (see\n        `Boolean arguments`). Notice that with Python 3.5 and earlier\n        dictionary order is undefined unless using ordered dictionaries.\n\n        Items are returned as a flat list so that first item is a key,\n        second item is a corresponding value, third item is the second key,\n        and so on.\n\n        The given ``dictionary`` is never altered by this keyword.\n\n        Example:\n        | ${sorted} =   | Get Dictionary Items | ${D3} |\n        | ${unsorted} = | Get Dictionary Items | ${D3} | sort_keys=False |\n        =>\n        | ${sorted} = ['a', 1, 'b', 2, 'c', 3]\n        | ${unsorted} = ['b', 2, 'a', 1, 'c', 3]    # Order depends on Python version.\n\n        ``sort_keys`` is a new option in Robot Framework 3.1.2. Earlier items\n        were always sorted based on keys.\n        "
        self._validate_dictionary(dictionary)
        keys = self.get_dictionary_keys(dictionary, sort_keys=sort_keys)
        return [i for key in keys for i in (key, dictionary[key])]

    def get_from_dictionary(self, dictionary, key, default=NOT_SET):
        if False:
            print('Hello World!')
        'Returns a value from the given ``dictionary`` based on the given ``key``.\n\n        If the given ``key`` cannot be found from the ``dictionary``, this\n        keyword fails. If optional ``default`` value is given, it will be\n        returned instead of failing.\n\n        The given dictionary is never altered by this keyword.\n\n        Example:\n        | ${value} = | Get From Dictionary | ${D3} | b |\n        =>\n        | ${value} = 2\n\n        Support for ``default`` is new in Robot Framework 6.0.\n        '
        self._validate_dictionary(dictionary)
        try:
            return dictionary[key]
        except KeyError:
            if default is not NOT_SET:
                return default
            raise RuntimeError(f"Dictionary does not contain key '{key}'.")

    def dictionary_should_contain_key(self, dictionary, key, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if ``key`` is not found from ``dictionary``.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_dictionary(dictionary)
        _verify_condition(key in dictionary, f"Dictionary does not contain key '{key}'.", msg)

    def dictionary_should_not_contain_key(self, dictionary, key, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails if ``key`` is found from ``dictionary``.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_dictionary(dictionary)
        _verify_condition(key not in dictionary, f"Dictionary contains key '{key}'.", msg)

    def dictionary_should_contain_item(self, dictionary, key, value, msg=None):
        if False:
            while True:
                i = 10
        'An item of ``key`` / ``value`` must be found in a ``dictionary``.\n\n        Value is converted to unicode for comparison.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_dictionary(dictionary)
        self.dictionary_should_contain_key(dictionary, key, msg)
        assert_equal(dictionary[key], value, msg or f"Value of dictionary key '{key}' does not match", values=not msg)

    def dictionary_should_contain_value(self, dictionary, value, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails if ``value`` is not found from ``dictionary``.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_dictionary(dictionary)
        _verify_condition(value in dictionary.values(), f"Dictionary does not contain value '{value}'.", msg)

    def dictionary_should_not_contain_value(self, dictionary, value, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if ``value`` is found from ``dictionary``.\n\n        Use the ``msg`` argument to override the default error message.\n        '
        self._validate_dictionary(dictionary)
        _verify_condition(value not in dictionary.values(), f"Dictionary contains value '{value}'.", msg)

    def dictionaries_should_be_equal(self, dict1, dict2, msg=None, values=True, ignore_keys=None):
        if False:
            for i in range(10):
                print('nop')
        "Fails if the given dictionaries are not equal.\n\n        First the equality of dictionaries' keys is checked and after that all\n        the key value pairs. If there are differences between the values, those\n        are listed in the error message. The types of the dictionaries do not\n        need to be same.\n\n        ``ignore_keys`` can be used to provide a list of keys to ignore in the\n        comparison. It can be an actual list or a Python list literal. This\n        option is new in Robot Framework 6.1.\n\n        Examples:\n        | Dictionaries Should Be Equal | ${dict} | ${expected} |\n        | Dictionaries Should Be Equal | ${dict} | ${expected} | ignore_keys=${ignored} |\n        | Dictionaries Should Be Equal | ${dict} | ${expected} | ignore_keys=['key1', 'key2'] |\n\n        See `Lists Should Be Equal` for more information about configuring\n        the error message with ``msg`` and ``values`` arguments.\n        "
        self._validate_dictionary(dict1)
        self._validate_dictionary(dict2, 2)
        if ignore_keys:
            if isinstance(ignore_keys, str):
                try:
                    ignore_keys = literal_eval(ignore_keys)
                except Exception:
                    raise ValueError("Converting 'ignore_keys' to a list failed: " + get_error_message())
            if not is_list_like(ignore_keys):
                raise ValueError(f"'ignore_keys' must be list-like, got {type_name(ignore_keys)}.")
            dict1 = {k: v for (k, v) in dict1.items() if k not in ignore_keys}
            dict2 = {k: v for (k, v) in dict2.items() if k not in ignore_keys}
        keys = self._keys_should_be_equal(dict1, dict2, msg, values)
        self._key_values_should_be_equal(keys, dict1, dict2, msg, values)

    def dictionary_should_contain_sub_dictionary(self, dict1, dict2, msg=None, values=True):
        if False:
            while True:
                i = 10
        'Fails unless all items in ``dict2`` are found from ``dict1``.\n\n        See `Lists Should Be Equal` for more information about configuring\n        the error message with ``msg`` and ``values`` arguments.\n        '
        self._validate_dictionary(dict1)
        self._validate_dictionary(dict2, 2)
        keys = self.get_dictionary_keys(dict2)
        diffs = ', '.join((str(k) for k in keys if k not in dict1))
        _verify_condition(not diffs, f'Following keys missing from first dictionary: {diffs}', msg, values)
        self._key_values_should_be_equal(keys, dict1, dict2, msg, values)

    def log_dictionary(self, dictionary, level='INFO'):
        if False:
            print('Hello World!')
        'Logs the size and contents of the ``dictionary`` using given ``level``.\n\n        Valid levels are TRACE, DEBUG, INFO (default), and WARN.\n\n        If you only want to log the size, use keyword `Get Length` from\n        the BuiltIn library.\n        '
        self._validate_dictionary(dictionary)
        logger.write('\n'.join(self._log_dictionary(dictionary)), level)

    def _log_dictionary(self, dictionary):
        if False:
            for i in range(10):
                print('nop')
        if not dictionary:
            yield 'Dictionary is empty.'
        elif len(dictionary) == 1:
            yield 'Dictionary has one item:'
        else:
            yield f'Dictionary size is {len(dictionary)} and it contains following items:'
        for key in self.get_dictionary_keys(dictionary):
            yield f'{key}: {dictionary[key]}'

    def _keys_should_be_equal(self, dict1, dict2, msg, values):
        if False:
            return 10
        keys1 = self.get_dictionary_keys(dict1)
        keys2 = self.get_dictionary_keys(dict2)
        miss1 = ', '.join((str(k) for k in keys2 if k not in dict1))
        miss2 = ', '.join((str(k) for k in keys1 if k not in dict2))
        error = []
        if miss1:
            error += [f'Following keys missing from first dictionary: {miss1}']
        if miss2:
            error += [f'Following keys missing from second dictionary: {miss2}']
        _verify_condition(not error, '\n'.join(error), msg, values)
        return keys1

    def _key_values_should_be_equal(self, keys, dict1, dict2, msg, values):
        if False:
            i = 10
            return i + 15
        diffs = '\n'.join(self._yield_dict_diffs(keys, dict1, dict2))
        _verify_condition(not diffs, f'Following keys have different values:\n{diffs}', msg, values)

    def _yield_dict_diffs(self, keys, dict1, dict2):
        if False:
            for i in range(10):
                print('nop')
        for key in keys:
            try:
                assert_equal(dict1[key], dict2[key], msg=f'Key {key}')
            except AssertionError as err:
                yield str(err)

    def _validate_dictionary(self, dictionary, position=1):
        if False:
            return 10
        if not is_dict_like(dictionary):
            raise TypeError(f'Expected argument {position} to be a dictionary or dictionary-like, got {type_name(dictionary)} instead.')

class Collections(_List, _Dictionary):
    """A library providing keywords for handling lists and dictionaries.

    ``Collections`` is Robot Framework's standard library that provides a
    set of keywords for handling Python lists and dictionaries. This
    library has keywords, for example, for modifying and getting
    values from lists and dictionaries (e.g. `Append To List`, `Get
    From Dictionary`) and for verifying their contents (e.g. `Lists
    Should Be Equal`, `Dictionary Should Contain Value`).

    == Table of contents ==

    %TOC%

    = Related keywords in BuiltIn =

    Following keywords in the BuiltIn library can also be used with
    lists and dictionaries:

    | = Keyword Name =             | = Applicable With = |
    | `Create List`                | lists |
    | `Create Dictionary`          | dicts |
    | `Get Length`                 | both  |
    | `Length Should Be`           | both  |
    | `Should Be Empty`            | both  |
    | `Should Not Be Empty`        | both  |
    | `Should Contain`             | both  |
    | `Should Not Contain`         | both  |
    | `Should Contain X Times`     | lists |
    | `Should Not Contain X Times` | lists |
    | `Get Count`                  | lists |

    = Using with list-like and dictionary-like objects =

    List keywords that do not alter the given list can also be used
    with tuples, and to some extend also with other iterables.
    `Convert To List` can be used to convert tuples and other iterables
    to Python ``list`` objects.

    Similarly dictionary keywords can, for most parts, be used with other
    mappings. `Convert To Dictionary` can be used if real Python ``dict``
    objects are needed.

    = Boolean arguments =

    Some keywords accept arguments that are handled as Boolean values true or
    false. If such an argument is given as a string, it is considered false if
    it is an empty string or equal to ``FALSE``, ``NONE``, ``NO``, ``OFF`` or
    ``0``, case-insensitively. Keywords verifying something that allow dropping
    actual and expected values from the possible error message also consider
    string ``no values`` to be false. Other strings are considered true
    regardless their value, and other argument types are tested using the same
    [http://docs.python.org/library/stdtypes.html#truth|rules as in Python].

    True examples:
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=True    | # Strings are generally true.    |
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=yes     | # Same as the above.             |
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=${TRUE} | # Python ``True`` is true.       |
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=${42}   | # Numbers other than 0 are true. |

    False examples:
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=False    | # String ``false`` is false.   |
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=no       | # Also string ``no`` is false. |
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=${EMPTY} | # Empty string is false.       |
    | `Should Contain Match` | ${list} | ${pattern} | case_insensitive=${FALSE} | # Python ``False`` is false.   |
    | `Lists Should Be Equal` | ${x}   | ${y} | Custom error | values=no values | # ``no values`` works with ``values`` argument |

    Considering ``OFF`` and ``0`` false is new in Robot Framework 3.1.

    = Data in examples =

    List related keywords use variables in format ``${Lx}`` in their examples.
    They mean lists with as many alphabetic characters as specified by ``x``.
    For example, ``${L1}`` means ``['a']`` and ``${L3}`` means
    ``['a', 'b', 'c']``.

    Dictionary keywords use similar ``${Dx}`` variables. For example, ``${D1}``
    means ``{'a': 1}`` and ``${D3}`` means ``{'a': 1, 'b': 2, 'c': 3}``.
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = get_version()

    def should_contain_match(self, list, pattern, msg=None, case_insensitive=False, whitespace_insensitive=False):
        if False:
            for i in range(10):
                print('nop')
        "Fails if ``pattern`` is not found in ``list``.\n\n        By default, pattern matching is similar to matching files in a shell\n        and is case-sensitive and whitespace-sensitive. In the pattern syntax,\n        ``*`` matches to anything and ``?`` matches to any single character. You\n        can also prepend ``glob=`` to your pattern to explicitly use this pattern\n        matching behavior.\n\n        If you prepend ``regexp=`` to your pattern, your pattern will be used\n        according to the Python\n        [http://docs.python.org/library/re.html|re module] regular expression\n        syntax. Important note: Backslashes are an escape character, and must\n        be escaped with another backslash (e.g. ``regexp=\\\\d{6}`` to search for\n        ``\\d{6}``). See `BuiltIn.Should Match Regexp` for more details.\n\n        If ``case_insensitive`` is given a true value (see `Boolean arguments`),\n        the pattern matching will ignore case.\n\n        If ``whitespace_insensitive`` is given a true value (see `Boolean\n        arguments`), the pattern matching will ignore whitespace.\n\n        Non-string values in lists are ignored when matching patterns.\n\n        Use the ``msg`` argument to override the default error message.\n\n        See also ``Should Not Contain Match``.\n\n        Examples:\n        | Should Contain Match | ${list} | a*              | | | # Match strings beginning with 'a'. |\n        | Should Contain Match | ${list} | regexp=a.*      | | | # Same as the above but with regexp. |\n        | Should Contain Match | ${list} | regexp=\\\\d{6} | | | # Match strings containing six digits. |\n        | Should Contain Match | ${list} | a*  | case_insensitive=True       | | # Match strings beginning with 'a' or 'A'. |\n        | Should Contain Match | ${list} | ab* | whitespace_insensitive=yes  | | # Match strings beginning with 'ab' with possible whitespace ignored. |\n        | Should Contain Match | ${list} | ab* | whitespace_insensitive=true | case_insensitive=true | # Same as the above but also ignore case. |\n        "
        _List._validate_list(self, list)
        matches = _get_matches_in_iterable(list, pattern, case_insensitive, whitespace_insensitive)
        default = f"{seq2str2(list)} does not contain match for pattern '{pattern}'."
        _verify_condition(matches, default, msg)

    def should_not_contain_match(self, list, pattern, msg=None, case_insensitive=False, whitespace_insensitive=False):
        if False:
            i = 10
            return i + 15
        'Fails if ``pattern`` is found in ``list``.\n\n        Exact opposite of `Should Contain Match` keyword. See that keyword\n        for information about arguments and usage in general.\n        '
        _List._validate_list(self, list)
        matches = _get_matches_in_iterable(list, pattern, case_insensitive, whitespace_insensitive)
        default = f"{seq2str2(list)} contains match for pattern '{pattern}'."
        _verify_condition(not matches, default, msg)

    def get_matches(self, list, pattern, case_insensitive=False, whitespace_insensitive=False):
        if False:
            for i in range(10):
                print('nop')
        "Returns a list of matches to ``pattern`` in ``list``.\n\n        For more information on ``pattern``, ``case_insensitive``, and\n        ``whitespace_insensitive``, see `Should Contain Match`.\n\n        Examples:\n        | ${matches}= | Get Matches | ${list} | a* | # ${matches} will contain any string beginning with 'a' |\n        | ${matches}= | Get Matches | ${list} | regexp=a.* | # ${matches} will contain any string beginning with 'a' (regexp version) |\n        | ${matches}= | Get Matches | ${list} | a* | case_insensitive=${True} | # ${matches} will contain any string beginning with 'a' or 'A' |\n        "
        _List._validate_list(self, list)
        return _get_matches_in_iterable(list, pattern, case_insensitive, whitespace_insensitive)

    def get_match_count(self, list, pattern, case_insensitive=False, whitespace_insensitive=False):
        if False:
            return 10
        "Returns the count of matches to ``pattern`` in ``list``.\n\n        For more information on ``pattern``, ``case_insensitive``, and\n        ``whitespace_insensitive``, see `Should Contain Match`.\n\n        Examples:\n        | ${count}= | Get Match Count | ${list} | a* | # ${count} will be the count of strings beginning with 'a' |\n        | ${count}= | Get Match Count | ${list} | regexp=a.* | # ${matches} will be the count of strings beginning with 'a' (regexp version) |\n        | ${count}= | Get Match Count | ${list} | a* | case_insensitive=${True} | # ${matches} will be the count of strings beginning with 'a' or 'A' |\n        "
        _List._validate_list(self, list)
        return len(self.get_matches(list, pattern, case_insensitive, whitespace_insensitive))

def _verify_condition(condition, default_msg, msg, values=False):
    if False:
        i = 10
        return i + 15
    if condition:
        return
    if not msg:
        msg = default_msg
    elif is_truthy(values) and str(values).upper() != 'NO VALUES':
        msg += '\n' + default_msg
    raise AssertionError(msg)

def _get_matches_in_iterable(iterable, pattern, case_insensitive=False, whitespace_insensitive=False):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(pattern, str):
        raise TypeError(f"Pattern must be string, got '{type_name(pattern)}'.")
    regexp = False
    if pattern.startswith('regexp='):
        pattern = pattern[7:]
        regexp = True
    elif pattern.startswith('glob='):
        pattern = pattern[5:]
    matcher = Matcher(pattern, caseless=is_truthy(case_insensitive), spaceless=is_truthy(whitespace_insensitive), regexp=regexp)
    return [item for item in iterable if isinstance(item, str) and matcher.match(item)]