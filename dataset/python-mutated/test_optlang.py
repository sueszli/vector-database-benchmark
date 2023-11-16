import pytest

def test_optlang(selenium):
    if False:
        print('Hello World!')
    selenium.load_package('optlang')
    selenium.run("\n        from optlang import Model, Variable, Constraint, Objective\n\n        # All the (symbolic) variables are declared, with a name and optionally a lower and/or upper bound.\n        x1 = Variable('x1', lb=0)\n        x2 = Variable('x2', lb=0)\n        x3 = Variable('x3', lb=0)\n\n        # A constraint is constructed from an expression of variables and a lower and/or upper bound (lb and ub).\n        c1 = Constraint(x1 + x2 + x3, ub=100)\n        c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)\n        c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)\n\n        # An objective can be formulated\n        obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')\n\n        # Variables, constraints and objective are combined in a Model object, which can subsequently be optimized.\n        model = Model(name='Simple model')\n        model.objective = obj\n        model.add([c1, c2, c3])\n\n        status = model.optimize()\n        ")
    result = selenium.run('model.status')
    assert result == 'optimal'
    result = selenium.run('model.objective.value')
    assert result == pytest.approx(733.3333, abs=0.0001)
    result = selenium.run("model.variables['x1'].primal")
    assert result == pytest.approx(33.3333, abs=0.0001)
    result = selenium.run("model.variables['x2'].primal")
    assert result == pytest.approx(66.6667, abs=0.0001)
    result = selenium.run("model.variables['x3'].primal")
    assert result == pytest.approx(0.0, abs=0.0001)