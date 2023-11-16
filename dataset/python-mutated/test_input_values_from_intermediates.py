from dagster import In, List, Optional, job, op

def test_from_intermediates_from_multiple_outputs():
    if False:
        for i in range(10):
            print('nop')

    @op
    def x():
        if False:
            return 10
        return 'x'

    @op
    def y():
        if False:
            return 10
        return 'y'

    @op(ins={'stuff': In(Optional[List[str]])})
    def gather(stuff):
        if False:
            while True:
                i = 10
        return '{} and {}'.format(*stuff)

    @job
    def pipe():
        if False:
            for i in range(10):
                print('nop')
        gather([x(), y()])
    result = pipe.execute_in_process()
    assert result
    assert result.success
    step_input_event = next((evt for evt in result.events_for_node('gather') if evt.event_type_value == 'STEP_INPUT' and evt.event_specific_data.input_name == 'stuff'))
    assert step_input_event.event_specific_data[1].label == 'stuff'
    assert result.output_for_node('gather') == 'x and y'

def test_from_intermediates_from_config():
    if False:
        for i in range(10):
            print('nop')
    run_config = {'ops': {'x': {'inputs': {'string_input': {'value': 'Dagster is great!'}}}}}

    @op
    def x(string_input):
        if False:
            i = 10
            return i + 15
        return string_input

    @job
    def pipe():
        if False:
            for i in range(10):
                print('nop')
        x()
    result = pipe.execute_in_process(run_config=run_config)
    assert result
    assert result.success
    step_input_event = next((evt for evt in result.events_for_node('x') if evt.event_type_value == 'STEP_INPUT' and evt.event_specific_data.input_name == 'string_input'))
    assert step_input_event.event_specific_data[1].label == 'string_input'
    assert result.output_for_node('x') == 'Dagster is great!'