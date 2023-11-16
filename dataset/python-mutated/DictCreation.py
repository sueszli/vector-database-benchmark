import itertools
module_value1 = module_value2 = module_value3 = module_value4 = 1000
module_key1 = module_key2 = module_key3 = module_key4 = 1000

def calledRepeatedly():
    if False:
        i = 10
        return i + 15
    dict_key1 = module_value1
    dict_key2 = module_value2
    dict_key3 = module_value3
    dict_key4 = module_value4
    dict_val1 = module_value1
    dict_val2 = module_value2
    dict_val3 = module_value3
    dict_val4 = module_value4
    l = {dict_key1: dict_val1, dict_key2: dict_val2, dict_key3: dict_val3, dict_key4: dict_val4}
    l = 1
    return (l, dict_val1, dict_val2, dict_val3, dict_val4, dict_key1, dict_key2, dict_key3, dict_key4)
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')