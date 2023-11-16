def testClassNamespaceOverridesClosure(self):
    if False:
        while True:
            i = 10
    x = 42

    class X:
        locals()['x'] = 43
        y = x