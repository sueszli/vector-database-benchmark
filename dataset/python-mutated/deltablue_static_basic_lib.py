"""
deltablue.py
============

Ported for the PyPy project.
Contributed by Daniel Lindsley

This implementation of the DeltaBlue benchmark was directly ported
from the `V8's source code`_, which was in turn derived
from the Smalltalk implementation by John Maloney and Mario
Wolczko. The original Javascript implementation was licensed under the GPL.

It's been updated in places to be more idiomatic to Python (for loops over
collections, a couple magic methods, ``OrderedCollection`` being a list & things
altering those collections changed to the builtin methods) but largely retains
the layout & logic from the original. (Ugh.)

.. _`V8's source code`: (https://github.com/v8/v8/blob/master/benchmarks/deltablue.js)
"""
from __future__ import annotations
import __static__
from __static__ import cast
from typing import final

class OrderedCollection(list):
    pass

@final
class Strength(object):
    REQUIRED = None
    STRONG_PREFERRED = None
    PREFERRED = None
    STRONG_DEFAULT = None
    NORMAL = None
    WEAK_DEFAULT = None
    WEAKEST = None

    def __init__(self, strength: int, name: str) -> None:
        if False:
            while True:
                i = 10
        super(Strength, self).__init__()
        self.strength = strength
        self.name = name

    @classmethod
    def stronger(cls, s1: Strength, s2: Strength) -> bool:
        if False:
            i = 10
            return i + 15
        return s1.strength < s2.strength

    @classmethod
    def weaker(cls, s1: Strength, s2: Strength) -> bool:
        if False:
            while True:
                i = 10
        return s1.strength > s2.strength

    @classmethod
    def weakest_of(cls, s1: Strength, s2: Strength) -> Strength:
        if False:
            while True:
                i = 10
        if cls.weaker(s1, s2):
            return s1
        return s2

    @classmethod
    def strongest(cls, s1, s2):
        if False:
            return 10
        if cls.stronger(s1, s2):
            return s1
        return s2

    def next_weaker(self) -> Strength:
        if False:
            return 10
        strengths = {0: self.__class__.WEAKEST, 1: self.__class__.WEAK_DEFAULT, 2: self.__class__.NORMAL, 3: self.__class__.STRONG_DEFAULT, 4: self.__class__.PREFERRED, 5: self.__class__.REQUIRED}
        return strengths[self.strength]
Strength.REQUIRED = Strength(0, 'required')
Strength.STRONG_PREFERRED = Strength(1, 'strongPreferred')
Strength.PREFERRED = Strength(2, 'preferred')
Strength.STRONG_DEFAULT = Strength(3, 'strongDefault')
Strength.NORMAL = Strength(4, 'normal')
Strength.WEAK_DEFAULT = Strength(5, 'weakDefault')
Strength.WEAKEST = Strength(6, 'weakest')

class Constraint(object):

    def __init__(self, strength: Strength) -> None:
        if False:
            print('Hello World!')
        super(Constraint, self).__init__()
        self.strength = strength

    def add_constraint(self) -> None:
        if False:
            i = 10
            return i + 15
        planner = get_planner()
        self.add_to_graph()
        planner.incremental_add(self)

    def satisfy(self, mark: int) -> Constraint | None:
        if False:
            i = 10
            return i + 15
        planner = get_planner()
        self.choose_method(mark)
        if not self.is_satisfied():
            if self.strength == Strength.REQUIRED:
                print('Could not satisfy a required constraint!')
            return None
        self.mark_inputs(mark)
        out = self.output()
        overridden = out.determined_by
        if overridden is not None:
            overridden.mark_unsatisfied()
        out.determined_by = self
        if not planner.add_propagate(self, mark):
            print('Cycle encountered')
        out.mark = mark
        return overridden

    def destroy_constraint(self) -> None:
        if False:
            return 10
        planner = get_planner()
        if self.is_satisfied():
            planner.incremental_remove(self)
        else:
            self.remove_from_graph()

    def is_input(self) -> bool:
        if False:
            print('Hello World!')
        return False

class UrnaryConstraint(Constraint):

    def __init__(self, v: Variable, strength: Strength) -> None:
        if False:
            i = 10
            return i + 15
        super(UrnaryConstraint, self).__init__(strength)
        self.my_output = v
        self.satisfied = False
        self.add_constraint()

    def add_to_graph(self) -> None:
        if False:
            while True:
                i = 10
        self.my_output.add_constraint(self)
        self.satisfied = False

    def choose_method(self, mark: int) -> None:
        if False:
            while True:
                i = 10
        if self.my_output.mark != mark and Strength.stronger(self.strength, self.my_output.walk_strength):
            self.satisfied = True
        else:
            self.satisfied = False

    def is_satisfied(self) -> bool:
        if False:
            print('Hello World!')
        return self.satisfied

    def mark_inputs(self, mark: int) -> None:
        if False:
            print('Hello World!')
        pass

    def output(self) -> Variable:
        if False:
            print('Hello World!')
        return self.my_output

    def recalculate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.my_output.walk_strength = self.strength
        self.my_output.stay = not self.is_input()
        if self.my_output.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        if False:
            while True:
                i = 10
        self.satisfied = False

    def inputs_known(self, mark: int) -> bool:
        if False:
            while True:
                i = 10
        return True

    def remove_from_graph(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.my_output is not None:
            self.my_output.remove_constraint(self)
            self.satisfied = False

@final
class StayConstraint(UrnaryConstraint):

    def __init__(self, v: Variable, string: Strength) -> None:
        if False:
            return 10
        super(StayConstraint, self).__init__(v, string)

    def execute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

@final
class EditConstraint(UrnaryConstraint):

    def __init__(self, v: Variable, string: Strength) -> None:
        if False:
            print('Hello World!')
        super(EditConstraint, self).__init__(v, string)

    def is_input(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def execute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

@final
class Direction(object):
    NONE = 0
    FORWARD = 1
    BACKWARD = -1

class BinaryConstraint(Constraint):

    def __init__(self, v1: Variable, v2: Variable, strength: Strength) -> None:
        if False:
            return 10
        super(BinaryConstraint, self).__init__(strength)
        self.v1 = v1
        self.v2 = v2
        self.direction = Direction.NONE
        self.add_constraint()

    def choose_method(self, mark: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.v1.mark == mark:
            if self.v2.mark != mark and Strength.stronger(self.strength, self.v2.walk_strength):
                self.direction = Direction.FORWARD
            else:
                self.direction = Direction.BACKWARD
        if self.v2.mark == mark:
            if self.v1.mark != mark and Strength.stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        if Strength.weaker(self.v1.walk_strength, self.v2.walk_strength):
            if Strength.stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        elif Strength.stronger(self.strength, self.v2.walk_strength):
            self.direction = Direction.FORWARD
        else:
            self.direction = Direction.BACKWARD

    def add_to_graph(self) -> None:
        if False:
            print('Hello World!')
        self.v1.add_constraint(self)
        self.v2.add_constraint(self)
        self.direction = Direction.NONE

    def is_satisfied(self) -> bool:
        if False:
            return 10
        return self.direction != Direction.NONE

    def mark_inputs(self, mark: int) -> None:
        if False:
            print('Hello World!')
        self.input().mark = mark

    def input(self) -> Variable:
        if False:
            print('Hello World!')
        if self.direction == Direction.FORWARD:
            return self.v1
        return self.v2

    def output(self) -> Variable:
        if False:
            i = 10
            return i + 15
        if self.direction == Direction.FORWARD:
            return self.v2
        return self.v1

    def recalculate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        ihn = self.input()
        out = self.output()
        out.walk_strength = Strength.weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay
        if out.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.direction = Direction.NONE

    def inputs_known(self, mark: int) -> bool:
        if False:
            return 10
        i = self.input()
        return i.mark == mark or i.stay or i.determined_by is None

    def remove_from_graph(self):
        if False:
            print('Hello World!')
        if self.v1 is not None:
            self.v1.remove_constraint(self)
        if self.v2 is not None:
            self.v2.remove_constraint(self)
        self.direction = Direction.NONE

@final
class ScaleConstraint(BinaryConstraint):

    def __init__(self, src: Variable, scale: Variable, offset: Variable, dest: Variable, strength: Strength) -> None:
        if False:
            print('Hello World!')
        self.direction = Direction.NONE
        self.scale = scale
        self.offset = offset
        super(ScaleConstraint, self).__init__(src, dest, strength)

    def add_to_graph(self) -> None:
        if False:
            print('Hello World!')
        super(ScaleConstraint, self).add_to_graph()
        self.scale.add_constraint(self)
        self.offset.add_constraint(self)

    def remove_from_graph(self):
        if False:
            print('Hello World!')
        super(ScaleConstraint, self).remove_from_graph()
        if self.scale is not None:
            self.scale.remove_constraint(self)
        if self.offset is not None:
            self.offset.remove_constraint(self)

    def mark_inputs(self, mark: int) -> None:
        if False:
            print('Hello World!')
        super(ScaleConstraint, self).mark_inputs(mark)
        self.scale.mark = mark
        self.offset.mark = mark

    def execute(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.direction == Direction.FORWARD:
            self.v2.value = self.v1.value * self.scale.value + self.offset.value
        else:
            self.v1.value = (self.v2.value - self.offset.value) / self.scale.value

    def recalculate(self) -> None:
        if False:
            return 10
        ihn = self.input()
        out = self.output()
        out.walk_strength = Strength.weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay and self.scale.stay and self.offset.stay
        if out.stay:
            self.execute()

@final
class EqualityConstraint(BinaryConstraint):

    def execute(self) -> None:
        if False:
            i = 10
            return i + 15
        self.output().value = self.input().value

@final
class Variable(object):

    def __init__(self, name: str, initial_value: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(Variable, self).__init__()
        self.name = name
        self.value = initial_value
        self.constraints = OrderedCollection()
        self.determined_by = None
        self.mark = 0
        self.walk_strength = Strength.WEAKEST
        self.stay = True

    def __repr__(self):
        if False:
            return 10
        return '<Variable: %s - %s>' % (self.name, self.value)

    def add_constraint(self, constraint: Constraint) -> None:
        if False:
            print('Hello World!')
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: EditConstraint) -> None:
        if False:
            i = 10
            return i + 15
        self.constraints.remove(constraint)
        if self.determined_by == constraint:
            self.determined_by = None

@final
class Planner(object):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super(Planner, self).__init__()
        self.current_mark = 0

    def incremental_add(self, constraint: Constraint) -> None:
        if False:
            while True:
                i = 10
        mark = self.new_mark()
        overridden = constraint.satisfy(mark)
        while overridden is not None:
            overridden = overridden.satisfy(mark)

    def incremental_remove(self, constraint: Constraint) -> None:
        if False:
            i = 10
            return i + 15
        out = constraint.output()
        constraint.mark_unsatisfied()
        constraint.remove_from_graph()
        unsatisfied = self.remove_propagate_from(out)
        strength = Strength.REQUIRED
        repeat = True
        while repeat:
            for u in unsatisfied:
                if u.strength == strength:
                    self.incremental_add(u)
                strength = strength.next_weaker()
            repeat = strength != Strength.WEAKEST

    def new_mark(self) -> int:
        if False:
            i = 10
            return i + 15
        x = self.current_mark + 1
        self.current_mark = x
        return self.current_mark

    def make_plan(self, sources: OrderedCollection) -> Plan:
        if False:
            for i in range(10):
                print('nop')
        mark = self.new_mark()
        plan = Plan()
        todo = sources
        while len(todo):
            c = todo.pop(0)
            if c.output().mark != mark and c.inputs_known(mark):
                plan.add_constraint(c)
                c.output().mark = mark
                self.add_constraints_consuming_to(c.output(), todo)
        return plan

    def extract_plan_from_constraints(self, constraints: OrderedCollection) -> Plan:
        if False:
            print('Hello World!')
        sources = OrderedCollection()
        x = len(constraints)
        i = 0
        while i < x:
            c = constraints[i]
            if c.is_input() and c.is_satisfied():
                sources.append(c)
            i = i + 1
        return self.make_plan(sources)

    def add_propagate(self, c: Constraint, mark: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        todo = OrderedCollection()
        todo.append(c)
        while len(todo):
            d = todo.pop(0)
            if d.output().mark == mark:
                self.incremental_remove(c)
                return False
            d.recalculate()
            self.add_constraints_consuming_to(d.output(), todo)
        return True

    def remove_propagate_from(self, out: Variable) -> OrderedCollection:
        if False:
            for i in range(10):
                print('nop')
        out.determined_by = None
        out.walk_strength = Strength.WEAKEST
        out.stay = True
        unsatisfied = OrderedCollection()
        todo = OrderedCollection()
        todo.append(out)
        while len(todo):
            v = todo.pop(0)
            cs = v.constraints
            x = len(cs)
            i = 0
            while i < x:
                c = cs[i]
                if not c.is_satisfied():
                    unsatisfied.append(c)
                i = i + 1
            determining = v.determined_by
            cs = v.constraints
            x = len(cs)
            i = 0
            while i < x:
                c = cs[i]
                if c != determining and c.is_satisfied():
                    c.recalculate()
                    todo.append(c.output())
                i = i + 1
        return unsatisfied

    def add_constraints_consuming_to(self, v: Variable, coll: OrderedCollection) -> None:
        if False:
            print('Hello World!')
        determining = v.determined_by
        cc = v.constraints
        x = len(cc)
        i = 0
        while i < x:
            c = cc[i]
            if c != determining and c.is_satisfied():
                coll.append(c)
            i = i + 1

@final
class Plan(object):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super(Plan, self).__init__()
        self.v = OrderedCollection()

    def add_constraint(self, c: Constraint) -> None:
        if False:
            i = 10
            return i + 15
        self.v.append(c)

    def __len__(self):
        if False:
            return 10
        return len(self.v)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        return self.v[index]

    def execute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        x = len(self.v)
        i = 0
        while i < x:
            c = self.v[i]
            c.execute()
            i = i + 1

def recreate_planner() -> Planner:
    if False:
        print('Hello World!')
    global planner
    planner = Planner()
    return planner

def get_planner() -> Planner:
    if False:
        while True:
            i = 10
    global planner
    return planner

def chain_test(n: int) -> None:
    if False:
        i = 10
        return i + 15
    '\n    This is the standard DeltaBlue benchmark. A long chain of equality\n    constraints is constructed with a stay constraint on one end. An\n    edit constraint is then added to the opposite end and the time is\n    measured for adding and removing this constraint, and extracting\n    and executing a constraint satisfaction plan. There are two cases.\n    In case 1, the added constraint is stronger than the stay\n    constraint and values must propagate down the entire length of the\n    chain. In case 2, the added constraint is weaker than the stay\n    constraint so it cannot be accomodated. The cost in this case is,\n    of course, very low. Typical situations lie somewhere between these\n    two extremes.\n    '
    planner = recreate_planner()
    prev: Variable | None = None
    first: Variable | None = None
    last: Variable | None = None
    i = 0
    end = n + 1
    while i < n + 1:
        name = 'v%s' % i
        v = Variable(name)
        if prev is not None:
            EqualityConstraint(prev, v, Strength.REQUIRED)
        if i == 0:
            first = v
        if i == n:
            last = v
        prev = v
        i = i + 1
    first = cast(Variable, first)
    last = cast(Variable, last)
    StayConstraint(last, Strength.STRONG_DEFAULT)
    edit = EditConstraint(first, Strength.PREFERRED)
    edits = OrderedCollection()
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    i = 0
    while i < 100:
        first.value = i
        plan.execute()
        if last.value != i:
            print('Chain test failed.')
        i = i + 1

def projection_test(n: int) -> None:
    if False:
        print('Hello World!')
    '\n    This test constructs a two sets of variables related to each\n    other by a simple linear transformation (scale and offset). The\n    time is measured to change a variable on either side of the\n    mapping and to change the scale and offset factors.\n    '
    planner = recreate_planner()
    scale = Variable('scale', 10)
    offset = Variable('offset', 1000)
    src: Variable | None = None
    dests = OrderedCollection()
    i = 0
    dst = Variable('dst%s' % 0, 0)
    while i < n:
        src = Variable('src%s' % i, i)
        dst = Variable('dst%s' % i, i)
        dests.append(dst)
        StayConstraint(src, Strength.NORMAL)
        ScaleConstraint(src, scale, offset, dst, Strength.REQUIRED)
        i = i + 1
    src = cast(Variable, src)
    change(src, 17)
    if dst.value != 1170:
        print('Projection 1 failed')
    change(dst, 1050)
    if src.value != 5:
        print('Projection 2 failed')
    change(scale, 5)
    i = 0
    while i < n - 1:
        if dests[i].value != i * 5 + 1000:
            print('Projection 3 failed')
        i = i + 1
    change(offset, 2000)
    i = 0
    while i < n - 1:
        if dests[i].value != i * 5 + 2000:
            print('Projection 4 failed')
        i = i + 1

def change(v: Variable, new_value: int) -> None:
    if False:
        return 10
    planner = get_planner()
    edit = EditConstraint(v, Strength.PREFERRED)
    edits = OrderedCollection()
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    i = 0
    while i < 10:
        v.value = new_value
        plan.execute()
        i = i + 1
    edit.destroy_constraint()
planner = None

def delta_blue(n: int) -> None:
    if False:
        i = 10
        return i + 15
    chain_test(n)
    projection_test(n)