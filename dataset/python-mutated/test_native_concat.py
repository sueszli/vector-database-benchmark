from __future__ import annotations
from ansible.playbook.conditional import Conditional
from ansible.template import Templar
from units.mock.loader import DictDataLoader

def test_cond_eval():
    if False:
        while True:
            i = 10
    fake_loader = DictDataLoader({})
    variables = {'foo': True}
    templar = Templar(loader=fake_loader, variables=variables)
    cond = Conditional(loader=fake_loader)
    cond.when = ['foo']
    with templar.set_temporary_context(jinja2_native=True):
        assert cond.evaluate_conditional(templar, variables)