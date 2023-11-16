"""
Step definitions for providing notes/hints.
The note steps explain what was important in the last few steps of
this scenario (for a test reader).
"""
from __future__ import absolute_import
from behave import step

@step(u'note that "{remark}"')
def step_note_that(context, remark):
    if False:
        for i in range(10):
            print('nop')
    '\n    Used as generic step that provides an additional remark/hint\n    and enhance the readability/understanding without performing any check.\n\n    .. code-block:: gherkin\n\n        Given that today is "April 1st"\n          But note that "April 1st is Fools day (and beware)"\n    '
    log = getattr(context, 'log', None)
    if log:
        log.info(u'NOTE: %s;' % remark)