from common import ds_local

def test_nop(ds_local):
    if False:
        print('Hello World!')
    df = ds_local
    column_names = df.column_names
    result = df.nop(column_names[1])
    assert result is None
    result = df.nop(column_names)
    assert result is None
    result = df.nop()
    assert result is None