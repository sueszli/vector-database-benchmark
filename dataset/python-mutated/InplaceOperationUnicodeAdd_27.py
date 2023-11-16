import itertools
module_value1 = 5
additiv_global = u'*' * 500

def calledRepeatedly():
    if False:
        print('Hello World!')
    module_value1
    s = u'2'
    additiv = additiv_global
    s += additiv
    s += u'lala'
    s += u'lala'
    s += u'lala'
    s += u'lala'
    s += u'lala'
    s += additiv
    return s
for x in itertools.repeat(None, 5000):
    calledRepeatedly()
print('OK.')