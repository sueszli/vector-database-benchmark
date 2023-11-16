"""
Provides a step registry and step decorators.
The step registry allows to match steps (model elements) with
step implementations (step definitions). This is necessary to execute steps.
"""
from __future__ import absolute_import
from behave.matchers import Match, make_matcher
from behave.textutil import text as _text
__all__ = ['given', 'when', 'then', 'step', 'Given', 'When', 'Then', 'Step']

class AmbiguousStep(ValueError):
    pass

class StepRegistry(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.steps = {'given': [], 'when': [], 'then': [], 'step': []}

    @staticmethod
    def same_step_definition(step, other_pattern, other_location):
        if False:
            i = 10
            return i + 15
        return step.pattern == other_pattern and step.location == other_location and (other_location.filename != '<string>')

    def add_step_definition(self, keyword, step_text, func):
        if False:
            print('Hello World!')
        step_location = Match.make_location(func)
        step_type = keyword.lower()
        step_text = _text(step_text)
        step_definitions = self.steps[step_type]
        for existing in step_definitions:
            if self.same_step_definition(existing, step_text, step_location):
                return
            elif existing.match(step_text):
                message = u'%s has already been defined in\n  existing step %s'
                new_step = u"@%s('%s')" % (step_type, step_text)
                existing.step_type = step_type
                existing_step = existing.describe()
                existing_step += u' at %s' % existing.location
                raise AmbiguousStep(message % (new_step, existing_step))
        step_definitions.append(make_matcher(func, step_text))

    def find_step_definition(self, step):
        if False:
            print('Hello World!')
        candidates = self.steps[step.step_type]
        more_steps = self.steps['step']
        if step.step_type != 'step' and more_steps:
            candidates = list(candidates)
            candidates += more_steps
        for step_definition in candidates:
            if step_definition.match(step.name):
                return step_definition
        return None

    def find_match(self, step):
        if False:
            print('Hello World!')
        candidates = self.steps[step.step_type]
        more_steps = self.steps['step']
        if step.step_type != 'step' and more_steps:
            candidates = list(candidates)
            candidates += more_steps
        for step_definition in candidates:
            result = step_definition.match(step.name)
            if result:
                return result
        return None

    def make_decorator(self, step_type):
        if False:
            while True:
                i = 10

        def decorator(step_text):
            if False:
                for i in range(10):
                    print('nop')

            def wrapper(func):
                if False:
                    for i in range(10):
                        print('nop')
                self.add_step_definition(step_type, step_text, func)
                return func
            return wrapper
        return decorator
registry = StepRegistry()

def setup_step_decorators(run_context=None, registry=registry):
    if False:
        print('Hello World!')
    if run_context is None:
        run_context = globals()
    for step_type in ('given', 'when', 'then', 'step'):
        step_decorator = registry.make_decorator(step_type)
        run_context[step_type.title()] = run_context[step_type] = step_decorator
setup_step_decorators()