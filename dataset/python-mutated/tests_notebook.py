from tqdm.notebook import tqdm as tqdm_notebook

def test_notebook_disabled_description():
    if False:
        print('Hello World!')
    'Test that set_description works for disabled tqdm_notebook'
    with tqdm_notebook(1, disable=True) as t:
        t.set_description('description')