from behave import given

@given(u'a person named "{name}"')
def step_given_person_with_name(ctx, name):
    if False:
        i = 10
        return i + 15
    pass