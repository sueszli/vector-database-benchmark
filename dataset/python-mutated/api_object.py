class ApiObject(object):
    """
        Filter count variable if the filter design tool has to
        return multiple filter parameters in future
        e.g Cascaded Filters
    """

    def __init__(self, filtcount=1):
        if False:
            return 10
        self.filtercount = filtcount
        self.restype = [''] * self.filtercount
        self.params = [''] * self.filtercount
        self.taps = [''] * self.filtercount
    '\n        Updates params dictionary for the given filter number\n    '

    def update_params(self, params, filtno):
        if False:
            for i in range(10):
                print('nop')
        if filtno <= self.filtercount:
            self.params[filtno - 1] = params
    '\n        Updates filter type  for the given filter number\n    '

    def update_filttype(self, filttype, filtno):
        if False:
            return 10
        if filtno <= self.filtercount:
            self.filttype[filtno - 1] = filttype
    '\n        updates taps for the given filter number. taps will\n        contain a list of coefficients in the case of fir design\n        and (b,a) tuple in the case of iir design\n    '

    def update_taps(self, taps, filtno):
        if False:
            return 10
        if filtno <= self.filtercount:
            self.taps[filtno - 1] = taps
    '\n        updates  all of them in a single call\n    '

    def update_all(self, filttype, params, taps, filtno):
        if False:
            i = 10
            return i + 15
        if filtno <= self.filtercount:
            self.taps[filtno - 1] = taps
            self.params[filtno - 1] = params
            self.restype[filtno - 1] = filttype

    def get_filtercount(self):
        if False:
            for i in range(10):
                print('nop')
        return self.filtercount

    def get_restype(self, filtno=1):
        if False:
            return 10
        if filtno <= self.filtercount:
            return self.restype[filtno - 1]

    def get_params(self, filtno=1):
        if False:
            print('Hello World!')
        if filtno <= self.filtercount:
            return self.params[filtno - 1]

    def get_taps(self, filtno=1):
        if False:
            for i in range(10):
                print('nop')
        if filtno <= self.filtercount:
            return self.taps[filtno - 1]