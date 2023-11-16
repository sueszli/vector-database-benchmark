from docs_snippets.intro_tutorial.basics.connecting_ops.complex_job import diamond

def test_complex_graph():
    if False:
        for i in range(10):
            print('nop')
    result = diamond.execute_in_process()
    assert result.success