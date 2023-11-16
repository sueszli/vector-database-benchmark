"""Objects to represent NEXUS standard data type matrix coding."""

class NexusError(Exception):
    """Provision for the management of Nexus exceptions."""

class StandardData:
    """Create a StandardData iterable object.

    Each coding specifies t [type] => (std [standard], multi [multistate] or
    uncer [uncertain]) and d [data]
    """

    def __init__(self, data):
        if False:
            print('Hello World!')
        'Initialize the class.'
        self._data = []
        self._current_pos = 0
        if not isinstance(data, str):
            raise NexusError('The coding data given to a StandardData object should be a string')
        multi_coding = False
        uncertain_coding = False
        coding_list = {'t': 'std', 'd': []}
        for (pos, coding) in enumerate(data):
            if multi_coding:
                if coding == ')':
                    multi_coding = False
                else:
                    coding_list['d'].append(coding)
                    continue
            elif uncertain_coding:
                if coding == '}':
                    uncertain_coding = False
                else:
                    coding_list['d'].append(coding)
                    continue
            elif coding == '(':
                multi_coding = True
                coding_list['t'] = 'multi'
                continue
            elif coding == '{':
                uncertain_coding = True
                coding_list['t'] = 'uncer'
                continue
            elif coding in [')', '}']:
                raise NexusError('Improper character %s at position %i of a coding sequence.' % (coding, pos))
            else:
                coding_list['d'].append(coding)
            self._data.append(coding_list.copy())
            coding_list = {'t': 'std', 'd': []}

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the length of the coding, use len(my_coding).'
        return len(self._data)

    def __getitem__(self, arg):
        if False:
            print('Hello World!')
        'Pull out child by index.'
        return self._data[arg]

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Iterate over the items.'
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        'Return next item.'
        try:
            return_coding = self._data[self._current_pos]
        except IndexError:
            self._current_pos = 0
            raise StopIteration from None
        else:
            self._current_pos += 1
            return return_coding

    def raw(self):
        if False:
            print('Hello World!')
        'Return the full coding as a python list.'
        return self._data

    def __str__(self):
        if False:
            print('Hello World!')
        'Return the full coding as a python string, use str(my_coding).'
        str_return = ''
        for coding in self._data:
            if coding['t'] == 'multi':
                str_return += '(' + ''.join(coding['d']) + ')'
            elif coding['t'] == 'uncer':
                str_return += '{' + ''.join(coding['d']) + '}'
            else:
                str_return += coding['d'][0]
        return str_return