def test_count_huge(client):
    if False:
        print('Hello World!')
    df = client['huge']
    assert df.count() == len(df)