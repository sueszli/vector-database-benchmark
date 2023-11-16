from dagster import graph, job, op

@op
def hello(_context):
    if False:
        print('Hello World!')
    return 'hello'

@op
def echo(_context, x):
    if False:
        return 10
    return x

def test_job_autoalias():
    if False:
        return 10

    @job
    def autopipe():
        if False:
            for i in range(10):
                print('nop')
        echo(echo(echo(hello())))
    result = autopipe.execute_in_process()
    assert result.success is True
    assert result.output_for_node('echo_3') == 'hello'
    assert result.output_for_node('echo_2') == 'hello'
    assert result.output_for_node('echo') == 'hello'
    assert result.output_for_node('hello') == 'hello'

def test_composite_autoalias():
    if False:
        return 10

    @graph
    def mega_echo(foo):
        if False:
            while True:
                i = 10
        echo(echo(echo(foo)))

    @job
    def autopipe():
        if False:
            i = 10
            return i + 15
        mega_echo(hello())
    result = autopipe.execute_in_process()
    assert result.success is True
    assert result.output_for_node('mega_echo.echo_3') == 'hello'
    assert result.output_for_node('mega_echo.echo_2') == 'hello'
    assert result.output_for_node('mega_echo.echo') == 'hello'
    assert result.output_for_node('hello') == 'hello'