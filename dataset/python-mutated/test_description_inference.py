from dagster import graph, job, op, resource

def test_description_inference():
    if False:
        print('Hello World!')
    decorators = [job, op, graph, resource]
    for decorator in decorators:

        @decorator
        def my_thing():
            if False:
                return 10
            'Here is some\n            multiline description.\n            '
        assert my_thing.description == '\n'.join(['Here is some', 'multiline description.'])