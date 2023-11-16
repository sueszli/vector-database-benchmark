import collections

class State:
    """
    record relationship of forward op/value and backward op/value
    one state must be bining with a program

    """

    def __init__(self, program):
        if False:
            i = 10
            return i + 15
        self.program = program
        self.value_to_valuegrad = collections.defaultdict(list)
        self.value_to_sumvaluegrad = collections.defaultdict(list)
        self.op_to_opgrad = collections.defaultdict(list)
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        self.opgrad_to_op = collections.defaultdict(list)

    def turn_map(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        self.opgrad_to_op = collections.defaultdict(list)
        for (k, v) in self.value_to_valuegrad.items():
            if v != []:
                for value in v[0]:
                    self.valuegrad_to_value[value] = [k]
        for (k, v) in self.value_to_sumvaluegrad.items():
            if v != []:
                for value in v[0]:
                    self.sumvaluegrad_to_value[value] = [k]
        for (k, v) in self.op_to_opgrad.items():
            if v != []:
                self.opgrad_to_op[v[0]] = [k]