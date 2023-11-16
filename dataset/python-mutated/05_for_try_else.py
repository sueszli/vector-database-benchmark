def test_iziplongest(self):
    if False:
        for i in range(10):
            print('nop')
    for args in ['abc']:
        self.assertEqual(1, 2)
    pass
    for stmt in ["izip_longest('abc', fv=1)"]:
        try:
            eval(stmt)
        except TypeError:
            pass
        else:
            self.fail()