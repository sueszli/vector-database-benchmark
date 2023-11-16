"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import abc
import cvxpy as cp

class Node:
    """ A node connecting devices. """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.voltage = cp.Variable()
        self.current_flows = []

    def constraints(self):
        if False:
            print('Hello World!')
        return [sum((f for f in self.current_flows)) == 0]

class Ground(Node):
    """ A node at 0 volts. """

    def constraints(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.voltage == 0] + super(Ground, self).constraints()

class Device:
    __metaclass__ = abc.ABCMeta
    ' A device on a circuit. '

    def __init__(self, pos_node, neg_node) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pos_node = pos_node
        self.pos_node.current_flows.append(-self.current())
        self.neg_node = neg_node
        self.neg_node.current_flows.append(self.current())

    @abc.abstractmethod
    def voltage(self):
        if False:
            return 10
        raise NotImplementedError()

    @abc.abstractmethod
    def current(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def constraints(self):
        if False:
            i = 10
            return i + 15
        return [self.pos_node.voltage - self.voltage() == self.neg_node.voltage]

class Resistor(Device):
    """ A resistor with V = R*I. """

    def __init__(self, pos_node, neg_node, resistance) -> None:
        if False:
            return 10
        self._current = cp.Variable()
        self.resistance = resistance
        super(Resistor, self).__init__(pos_node, neg_node)

    def voltage(self):
        if False:
            i = 10
            return i + 15
        return -self.resistance * self.current()

    def current(self):
        if False:
            return 10
        return self._current

class VoltageSource(Device):
    """ A constant source of voltage. """

    def __init__(self, pos_node, neg_node, voltage) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._current = cp.Variable()
        self._voltage = voltage
        super(VoltageSource, self).__init__(pos_node, neg_node)

    def voltage(self):
        if False:
            return 10
        return self._voltage

    def current(self):
        if False:
            return 10
        return self._current

class CurrentSource(Device):
    """ A constant source of current. """

    def __init__(self, pos_node, neg_node, current) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._current = current
        self._voltage = cp.Variable()
        super(CurrentSource, self).__init__(pos_node, neg_node)

    def voltage(self):
        if False:
            return 10
        return self._voltage

    def current(self):
        if False:
            return 10
        return self._current
nodes = [Ground(), Node(), Node()]
devices = [VoltageSource(nodes[0], nodes[2], 10)]
devices.append(Resistor(nodes[0], nodes[1], 0.25))
devices.append(Resistor(nodes[0], nodes[1], 1))
devices.append(Resistor(nodes[1], nodes[2], 4))
devices.append(Resistor(nodes[1], nodes[2], 1))
constraints = []
for obj in nodes + devices:
    constraints += obj.constraints()
cp.Problem(cp.Minimize(0), constraints).solve()
for node in nodes:
    print(node.voltage.value)