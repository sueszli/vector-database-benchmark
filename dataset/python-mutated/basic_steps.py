from behave import given, step

@step(u'{word} background step passes')
def step_background_step_passes(context, word):
    if False:
        while True:
            i = 10
    pass

@given(u'I need {word} scenario setup')
def step_given_i_need_scenario_setup(context, word):
    if False:
        for i in range(10):
            print('nop')
    pass