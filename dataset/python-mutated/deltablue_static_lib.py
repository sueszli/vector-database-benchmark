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
from __static__ import CheckedList, box, cast, cbool, clen, int64, inline
from typing import final
from enum import IntEnum

@inline
def stronger(s1: Strength, s2: Strength) -> cbool:
    if False:
        while True:
            i = 10
    return s1.strength < s2.strength

@inline
def weaker(s1: Strength, s2: Strength) -> cbool:
    if False:
        for i in range(10):
            print('nop')
    return s1.strength > s2.strength

@inline
def weakest_of(s1: Strength, s2: Strength) -> Strength:
    if False:
        print('Hello World!')
    return s1 if s1.strength > s2.strength else s2

@final
class Strength:

    def __init__(self, strength: int64, name: str) -> None:
        if False:
            return 10
        self.strength: int64 = strength
        self.name: str = name

    def next_weaker(self) -> Strength:
        if False:
            for i in range(10):
                print('nop')
        return STRENGTHS[self.strength]
REQUIRED = Strength(0, 'required')
STRONG_PREFERRED = Strength(1, 'strongPreferred')
PREFERRED = Strength(2, 'preferred')
STRONG_DEFAULT = Strength(3, 'strongDefault')
NORMAL = Strength(4, 'normal')
WEAK_DEFAULT = Strength(5, 'weakDefault')
WEAKEST = Strength(6, 'weakest')
STRENGTHS: CheckedList[Strength] = CheckedList[Strength]([WEAKEST, WEAK_DEFAULT, NORMAL, STRONG_DEFAULT, PREFERRED, REQUIRED])

class Constraint(object):

    def __init__(self, strength: Strength) -> None:
        if False:
            return 10
        self.strength: Strength = strength

    def add_constraint(self) -> None:
        if False:
            while True:
                i = 10
        planner = get_planner()
        self.add_to_graph()
        planner.incremental_add(self)

    def satisfy(self, mark: int64) -> Constraint | None:
        if False:
            return 10
        planner = get_planner()
        self.choose_method(mark)
        if not self.is_satisfied():
            if self.strength == REQUIRED:
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
            i = 10
            return i + 15
        planner = get_planner()
        if self.is_satisfied():
            planner.incremental_remove(self)
        else:
            self.remove_from_graph()

    def is_input(self) -> cbool:
        if False:
            while True:
                i = 10
        return False

    def mark_inputs(self, mark: int64) -> None:
        if False:
            print('Hello World!')
        pass

    def inputs_known(self, mark: int64) -> cbool:
        if False:
            while True:
                i = 10
        return True

    def choose_method(self, mark: int64) -> None:
        if False:
            while True:
                i = 10
        pass

    def output(self) -> Variable:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def execute(self) -> None:
        if False:
            print('Hello World!')
        pass

class UrnaryConstraint(Constraint):

    def __init__(self, v: Variable, strength: Strength) -> None:
        if False:
            return 10
        Constraint.__init__(self, strength)
        self.my_output: Variable = v
        self.satisfied: cbool = False
        self.add_constraint()

    def add_to_graph(self) -> None:
        if False:
            while True:
                i = 10
        self.my_output.add_constraint(self)
        self.satisfied = False

    def choose_method(self, mark: int64) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.my_output.mark != mark and stronger(self.strength, self.my_output.walk_strength):
            self.satisfied = True
        else:
            self.satisfied = False

    def is_satisfied(self) -> cbool:
        if False:
            while True:
                i = 10
        return self.satisfied

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
            return 10
        self.satisfied = False

    def remove_from_graph(self) -> None:
        if False:
            return 10
        if self.my_output is not None:
            self.my_output.remove_constraint(self)
            self.satisfied = False

@final
class StayConstraint(UrnaryConstraint):
    pass

@final
class EditConstraint(UrnaryConstraint):

    def is_input(self) -> cbool:
        if False:
            while True:
                i = 10
        return True

class Direction(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = -1

class BinaryConstraint(Constraint):

    def __init__(self, v1: Variable, v2: Variable, strength: Strength) -> None:
        if False:
            return 10
        Constraint.__init__(self, strength)
        self.v1: Variable = v1
        self.v2: Variable = v2
        self.direction: Direction = Direction.NONE
        self.add_constraint()

    def choose_method(self, mark: int64) -> None:
        if False:
            return 10
        if self.v1.mark == mark:
            if self.v2.mark != mark and stronger(self.strength, self.v2.walk_strength):
                self.direction = Direction.FORWARD
            else:
                self.direction = Direction.BACKWARD
        if self.v2.mark == mark:
            if self.v1.mark != mark and stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        if weaker(self.v1.walk_strength, self.v2.walk_strength):
            if stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        elif stronger(self.strength, self.v2.walk_strength):
            self.direction = Direction.FORWARD
        else:
            self.direction = Direction.BACKWARD

    def add_to_graph(self) -> None:
        if False:
            i = 10
            return i + 15
        self.v1.add_constraint(self)
        self.v2.add_constraint(self)
        self.direction = Direction.NONE

    def is_satisfied(self) -> cbool:
        if False:
            print('Hello World!')
        if self.direction != Direction.NONE:
            return True
        return False

    def mark_inputs(self, mark: int64) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.input().mark = mark

    def input(self) -> Variable:
        if False:
            return 10
        return self.v1 if self.direction == Direction.FORWARD else self.v2

    def output(self) -> Variable:
        if False:
            return 10
        return self.v2 if self.direction == Direction.FORWARD else self.v1

    def recalculate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        ihn = self.input()
        out = self.output()
        out.walk_strength = weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay
        if out.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.direction = Direction.NONE

    def inputs_known(self, mark: int64) -> cbool:
        if False:
            i = 10
            return i + 15
        i = self.input()
        return i.mark == mark or i.stay or cbool(i.determined_by is None)

    def remove_from_graph(self):
        if False:
            i = 10
            return i + 15
        if self.v1 is not None:
            self.v1.remove_constraint(self)
        if self.v2 is not None:
            self.v2.remove_constraint(self)
        self.direction = Direction.NONE

@final
class ScaleConstraint(BinaryConstraint):

    def __init__(self, src: Variable, scale: Variable, offset: Variable, dest: Variable, strength: Strength) -> None:
        if False:
            while True:
                i = 10
        self.direction: Direction = Direction.NONE
        self.scale: Variable = scale
        self.offset: Variable = offset
        BinaryConstraint.__init__(self, src, dest, strength)

    def add_to_graph(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        BinaryConstraint.add_to_graph(self)
        self.scale.add_constraint(self)
        self.offset.add_constraint(self)

    def remove_from_graph(self):
        if False:
            return 10
        BinaryConstraint.remove_from_graph(self)
        if self.scale is not None:
            self.scale.remove_constraint(self)
        if self.offset is not None:
            self.offset.remove_constraint(self)

    def mark_inputs(self, mark: int64) -> None:
        if False:
            while True:
                i = 10
        BinaryConstraint.mark_inputs(self, mark)
        self.scale.mark = mark
        self.offset.mark = mark

    def execute(self) -> None:
        if False:
            return 10
        if self.direction == Direction.FORWARD:
            self.v2.value = self.v1.value * self.scale.value + self.offset.value
        else:
            self.v1.value = (self.v2.value - self.offset.value) / self.scale.value

    def recalculate(self) -> None:
        if False:
            print('Hello World!')
        ihn = self.input()
        out = self.output()
        out.walk_strength = weakest_of(self.strength, ihn.walk_strength)
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

    def __init__(self, name: str, initial_value: int64=0) -> None:
        if False:
            print('Hello World!')
        self.name: str = name
        self.value: int64 = initial_value
        self.constraints: CheckedList[Constraint] = []
        self.determined_by: Constraint | None = None
        self.mark: int64 = 0
        self.walk_strength: Strength = WEAKEST
        self.stay: cbool = True

    def add_constraint(self, constraint: Constraint) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: Constraint) -> None:
        if False:
            return 10
        self.constraints.remove(constraint)
        if self.determined_by == constraint:
            self.determined_by = None

@final
class Planner(object):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.current_mark: int64 = 0

    def incremental_add(self, constraint: Constraint) -> None:
        if False:
            for i in range(10):
                print('nop')
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
        strength = REQUIRED
        repeat = True
        while repeat:
            for u in unsatisfied:
                if u.strength == strength:
                    self.incremental_add(u)
                strength = strength.next_weaker()
            repeat = strength != WEAKEST

    def new_mark(self) -> int64:
        if False:
            print('Hello World!')
        x = self.current_mark + 1
        self.current_mark = x
        return self.current_mark

    def make_plan(self, sources: CheckedList[UrnaryConstraint]) -> Plan:
        if False:
            return 10
        mark = self.new_mark()
        plan = Plan()
        todo: CheckedList[Constraint] = [s for s in sources]
        while clen(todo):
            c = todo.pop(0)
            if c.output().mark != mark and c.inputs_known(mark):
                plan.add_constraint(c)
                c.output().mark = mark
                self.add_constraints_consuming_to(c.output(), todo)
        return plan

    def extract_plan_from_constraints(self, constraints: CheckedList[UrnaryConstraint]) -> Plan:
        if False:
            for i in range(10):
                print('nop')
        sources: CheckedList[UrnaryConstraint] = []
        for c in constraints:
            if c.is_input() and c.is_satisfied():
                sources.append(c)
        return self.make_plan(sources)

    def add_propagate(self, c: Constraint, mark: int64) -> cbool:
        if False:
            while True:
                i = 10
        todo: CheckedList[Constraint] = []
        todo.append(c)
        while clen(todo):
            d = todo.pop(0)
            if d.output().mark == mark:
                self.incremental_remove(c)
                return False
            d.recalculate()
            self.add_constraints_consuming_to(d.output(), todo)
        return True

    def remove_propagate_from(self, out: Variable) -> CheckedList[Constraint]:
        if False:
            return 10
        out.determined_by = None
        out.walk_strength = WEAKEST
        out.stay = True
        unsatisfied: CheckedList[Constraint] = []
        todo: CheckedList[Variable] = []
        todo.append(out)
        while len(todo):
            v = todo.pop(0)
            cs = v.constraints
            for c in cs:
                if not c.is_satisfied():
                    unsatisfied.append(c)
            determining = v.determined_by
            for c in cs:
                if c != determining and c.is_satisfied():
                    c.recalculate()
                    todo.append(c.output())
        return unsatisfied

    def add_constraints_consuming_to(self, v: Variable, coll: CheckedList[Constraint]) -> None:
        if False:
            return 10
        determining = v.determined_by
        cc = v.constraints
        for c in cc:
            if c != determining and c.is_satisfied():
                coll.append(c)

@final
class Plan(object):

    def __init__(self) -> None:
        if False:
            return 10
        self.v: CheckedList[Constraint] = []

    def add_constraint(self, c: Constraint) -> None:
        if False:
            return 10
        self.v.append(c)

    def __len__(self):
        if False:
            return 10
        return len(self.v)

    def __getitem__(self, index):
        if False:
            return 10
        return self.v[index]

    def execute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for c in self.v:
            c.execute()

def recreate_planner() -> Planner:
    if False:
        return 10
    global planner
    planner = Planner()
    return planner

def get_planner() -> Planner:
    if False:
        print('Hello World!')
    global planner
    return planner

def chain_test(n: int64) -> None:
    if False:
        i = 10
        return i + 15
    '\n    This is the standard DeltaBlue benchmark. A long chain of equality\n    constraints is constructed with a stay constraint on one end. An\n    edit constraint is then added to the opposite end and the time is\n    measured for adding and removing this constraint, and extracting\n    and executing a constraint satisfaction plan. There are two cases.\n    In case 1, the added constraint is stronger than the stay\n    constraint and values must propagate down the entire length of the\n    chain. In case 2, the added constraint is weaker than the stay\n    constraint so it cannot be accomodated. The cost in this case is,\n    of course, very low. Typical situations lie somewhere between these\n    two extremes.\n    '
    planner = recreate_planner()
    prev: Variable | None = None
    first: Variable | None = None
    last: Variable | None = None
    i: int64 = 0
    end: int64 = n + 1
    while i < n + 1:
        name = 'v%s' % box(i)
        v = Variable(name)
        if prev is not None:
            EqualityConstraint(prev, v, REQUIRED)
        if i == 0:
            first = v
        if i == n:
            last = v
        prev = v
        i = i + 1
    first = cast(Variable, first)
    last = cast(Variable, last)
    StayConstraint(last, STRONG_DEFAULT)
    edit = EditConstraint(first, PREFERRED)
    edits: CheckedList[UrnaryConstraint] = []
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    i = 0
    while i < 100:
        first.value = i
        plan.execute()
        if last.value != i:
            print('Chain test failed.')
        i = i + 1

def projection_test(n: int64) -> None:
    if False:
        print('Hello World!')
    '\n    This test constructs a two sets of variables related to each\n    other by a simple linear transformation (scale and offset). The\n    time is measured to change a variable on either side of the\n    mapping and to change the scale and offset factors.\n    '
    planner = recreate_planner()
    scale = Variable('scale', 10)
    offset = Variable('offset', 1000)
    src: Variable | None = None
    dests: CheckedList[Variable] = []
    i: int64 = 0
    bi = box(i)
    dst = Variable('dst%s' % bi, i)
    while i < n:
        bi = box(i)
        src = Variable('src%s' % bi, i)
        dst = Variable('dst%s' % bi, i)
        dests.append(dst)
        StayConstraint(src, NORMAL)
        ScaleConstraint(src, scale, offset, dst, REQUIRED)
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

def change(v: Variable, new_value: int64) -> None:
    if False:
        for i in range(10):
            print('nop')
    planner = get_planner()
    edit = EditConstraint(v, PREFERRED)
    edits: CheckedList[UrnaryConstraint] = []
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    i: int64 = 0
    while i < 10:
        v.value = new_value
        plan.execute()
        i = i + 1
    edit.destroy_constraint()
planner = None

def delta_blue(n: int64) -> None:
    if False:
        for i in range(10):
            print('nop')
    chain_test(n)
    projection_test(n)