def testClassNamespaceOverridesClosure(self):
    if False:
        return 10
    x = 42

    class X:
        locals()['x'] = 43
        del x